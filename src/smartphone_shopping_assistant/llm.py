"""Bounded, local Ollama intent parsing and candidate selection."""

from __future__ import annotations

import json
import re
import time
from typing import Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import ParsedIntent, SearchCandidate

MODEL = "qwen3:1.7b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
REASON_CODES = ("price", "brand", "memory", "unlocked", "semantic_match")
_AMOUNT = re.compile(r"(?<![\w.])(?P<value>\d+)\s*(?P<unit>gb|гб)?\b", re.I)
_RAM = re.compile(r"\bram\b|\bозу\b|оперативн\w*", re.I)
_STORAGE = re.compile(
    r"\bstorage\b|\brom\b|накопител\w*|хранилищ\w*|встроенн\w*\s+памят\w*|памят\w*",
    re.I,
)
_CYRILLIC = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})


def _memory_evidence(query: str) -> dict[str, dict[str, int | bool]]:
    """Extract only GB amounts with a defensible memory type and bound direction."""
    result: dict[str, dict[str, int | bool]] = {}
    matches = list(_AMOUNT.finditer(query))
    for index, match in enumerate(matches):
        value = int(match.group("value"))
        if not 0 < value <= 2_147_483_647:
            continue
        start = matches[index - 1].end() if index else 0
        before = query[max(start, match.start() - 45):match.start()].lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        after = query[match.end():min(end, match.end() + 20)].lower()
        before_ram, before_storage = bool(_RAM.search(before)), bool(_STORAGE.search(before))
        after_ram, after_storage = bool(_RAM.search(after)), bool(_STORAGE.search(after))
        next_bound = r"[<>≥≤]|\b(?:minimum|maximum|at least|at most|не менее|не более)\b"
        if re.match(r"\s*(?:and|и)\b", after) or re.search(next_bound, after):
            after_ram = after_storage = False
        # A following label wins for "8 GB RAM and 128 GB storage"; otherwise
        # use the label nearest the amount, never a label across another number.
        if after_ram != after_storage:
            field = "ram_gb" if after_ram else "storage_gb"
        elif before_ram != before_storage:
            field = "ram_gb" if before_ram else "storage_gb"
        elif match.group("unit") and value >= 32:
            field = "storage_gb"
        else:
            continue
        near = before[-35:].strip()
        lower = (
            r"(?:>=|≥|at\s+least|not\s+less\s+than|minimum|min\.?|"
            r"не\s+менее|как\s+минимум|от)\s*(?:\w+\s*){0,2}$"
        )
        upper = (
            r"(?:<=|≤|at\s+most|no\s+more\s+than|maximum|max\.?|"
            r"не\s+более|не\s+больше|до|up\s+to)\s*(?:\w+\s*){0,2}$"
        )
        if re.search(lower, near):
            bound = {"minimum": value, "include_minimum": True}
        elif re.search(upper, near):
            bound = {"maximum": value, "include_maximum": True}
        elif re.search(r"(?:<|under|less\s+than|below|менее|меньше|ниже)\s*(?:\w+\s*){0,2}$", near):
            bound = {"maximum": value, "include_maximum": False}
        elif re.search(r"(?:>|over|more\s+than|above|более|больше|выше)\s*(?:\w+\s*){0,2}$", near):
            bound = {"minimum": value, "include_minimum": False}
        else:
            bound = {"minimum": value, "maximum": value}
        if field in result and result[field] != bound:
            # Multiple incompatible amounts need clarification, not a guessed patch.
            raise ValueError("ambiguous memory amounts")
        result[field] = bound
    return result


class LLMUnavailable(RuntimeError):
    """The mandatory local model stage failed; details stay out of user messages."""


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_ids: list[UUID] = Field(default_factory=list, max_length=5)
    abstain: bool
    reason_codes: list[Literal["price", "brand", "memory", "unlocked", "semantic_match"]] = (
        Field(default_factory=list)
    )

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("duplicate selected ID")
        if self.abstain != (not self.selected_ids):
            raise ValueError("abstain must match empty selection")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("duplicate reason code")
        return self


class LocalLLM:
    """Use a caller-supplied monotonic deadline to share a 120s request budget.

    `parse` returns ParsedIntent. `select` returns Selection; its IDs are checked
    against the supplied SearchCandidate allowlist, but callers must still check
    current exact filters before presenting database-backed product facts.
    """

    def __init__(self, model: str = MODEL, client: httpx.Client | None = None):
        self.model = model
        self.client = client or httpx.Client(trust_env=False)

    def _chat(self, system: str, data: dict, schema: dict, deadline: float | None) -> dict:
        cutoff = deadline if deadline is not None else time.monotonic() + 120
        remaining = cutoff - time.monotonic()
        if remaining <= 0:
            raise LLMUnavailable("Local LLM deadline exceeded")
        try:
            response = self.client.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": schema,
                    "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 384},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
                    ],
                },
                timeout=min(remaining, 60),
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            answer = json.loads(content)
            if not isinstance(answer, dict):
                raise ValueError("object required")
            return answer
        except (httpx.HTTPError, KeyError, TypeError) as exc:
            raise LLMUnavailable("Local LLM unavailable") from exc

    def parse(
        self, query: str, contextual_info: dict | None = None, deadline: float | None = None
    ) -> ParsedIntent:
        if not query.strip() or len(query) > 500:
            raise ValueError("query must contain 1-500 characters")
        system = (
            "Parse Russian or English smartphone requests as JSON. Intent: search, refine, "
            "compare, clarify. For cheaper/lower price use refine with semantic_query "
            "describing cheaper products; do not invent a numeric price. "
            "filter_changes may contain brands, price, ram_gb, storage_gb, carrier_unlocked. "
            "Omit unchanged filter fields. To clear a filter, put its name in clear_fields "
            "and set that filter_changes field to null. Null without clear_fields is ignored. "
            "Ranges use minimum/maximum with inclusive flags. "
            "Plain '8 GB RAM' and '128 GB storage' mean exact values: set both minimum and "
            "maximum. 'At least', 'не менее', '>=' mean inclusive minimum only; 'under', "
            "'менее', '<' mean exclusive maximum only; 'at most', 'не более', '<=' mean "
            "inclusive maximum only. A plain 128GB means exact storage. "
            "Extract previous_result_positions only for explicit result numbers. "
            "If user requests unsupported attributes such as "
            "camera or battery, or currency conversion, use clarify. No SQL, tools, or facts."
        )
        deadline = deadline if deadline is not None else time.monotonic() + 120
        data = {"query": query, "context": contextual_info or {}}
        schema = ParsedIntent.model_json_schema()
        schema["properties"]["clear_fields"] = {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["brands", "price", "ram_gb", "storage_gb", "carrier_unlocked"],
            },
        }
        for attempt in range(2):
            try:
                answer = self._chat(system, data, schema, deadline)
                clear_fields = answer.pop("clear_fields", [])
                if not isinstance(clear_fields, list) or any(
                    field not in {"brands", "price", "ram_gb", "storage_gb", "carrier_unlocked"}
                    for field in clear_fields
                ):
                    raise ValueError("invalid clear_fields")
                if not re.search(r"убер\w*|сним\w*|очист\w*|clear|remove|reset", query.lower()):
                    clear_fields = []
                changes = answer.get("filter_changes", {})
                if isinstance(changes, dict):
                    answer["filter_changes"] = {
                        key: value for key, value in changes.items()
                        if value is not None or key in clear_fields
                    }
                    for field in clear_fields:
                        answer["filter_changes"][field] = None
                position_pattern = r"\b(?:[1-9]\d?|first|second|third|перв\w*|втор\w*|трет\w*)\b"
                query_lower = query.lower()
                if not re.search(position_pattern, query_lower):
                    answer["previous_result_positions"] = []
                elif re.search(r"перв\w*\s+два\b|first\s+two\b", query_lower):
                    answer["previous_result_positions"] = [1, 2]
                changes = answer.get("filter_changes", {})
                if isinstance(changes, dict):
                    try:
                        evidence = _memory_evidence(query)
                    except ValueError:
                        answer["intent"] = "clarify"
                        evidence = {}
                    for field in ("ram_gb", "storage_gb"):
                        if field in evidence and field not in clear_fields:
                            changes[field] = evidence[field]
                        elif field in changes and field not in clear_fields:
                            changes.pop(field)
                    brands = changes.get("brands")
                    if isinstance(brands, list) and any(
                        not isinstance(brand, str) or (
                            not re.search(
                                r"(?<!\w)" + re.escape(brand.casefold()) + r"(?!\w)",
                                query_lower,
                            )
                            and not re.search(
                                r"(?<!\w)" + re.escape(brand.casefold()) + r"(?!\w)",
                                query_lower.translate(_CYRILLIC),
                            )
                        )
                        for brand in brands
                    ):
                        changes.pop("brands", None)
                        answer["intent"] = "clarify"
                    if "price" in changes and "price" not in clear_fields and not re.search(
                        r"\d", query_lower
                    ):
                        changes.pop("price", None)
                    if not re.search(r"разблок|без\s+привяз|unlocked|carrier", query_lower):
                        changes.pop("carrier_unlocked", None)
                if answer.get("previous_result_positions") and re.search(
                    r"сравн\w*|compare", query_lower
                ):
                    answer["intent"] = "compare"
                return ParsedIntent.model_validate(answer)
            except (ValidationError, ValueError) as exc:
                if attempt:
                    raise LLMUnavailable("Invalid local LLM intent") from exc
        raise LLMUnavailable("Invalid local LLM intent")

    def select(
        self,
        query: str,
        candidates: list[SearchCandidate],
        limit: int = 5,
        deadline: float | None = None,
    ) -> Selection:
        if (
            not 1 <= limit <= 5
            or len(candidates) > 10
            or len({c.product.id for c in candidates}) != len(candidates)
        ):
            raise ValueError("invalid candidate count, IDs, or limit")
        if not candidates:
            return Selection(selected_ids=[], abstain=True, reason_codes=[])
        deadline = deadline if deadline is not None else time.monotonic() + 120
        allowed = {c.product.id for c in candidates}
        data = {
            "query": query,
            "limit": limit,
            "candidates": [
                {
                    "id": str(c.product.id),
                    "name": c.product.name,
                    "brand": c.product.brand,
                    "model": c.product.model,
                    "price": str(c.product.price),
                    "ram_gb": c.product.ram_gb,
                    "storage_gb": c.product.storage_gb,
                    "carrier_unlocked": c.product.carrier_unlocked,
                }
                for c in candidates
            ],
        }
        system = (
            "Choose at most limit smartphone IDs from candidates only, ordered by relevance. "
            "Return selected_ids, abstain, and reason_codes. Set abstain=false whenever "
            "selected_ids is nonempty; set abstain=true only when selected_ids is empty. "
            "Allowed reason codes: "
            + ", ".join(REASON_CODES)
            + ". No free-text reasons or invented facts. Empty selection is allowed."
        )
        for attempt in range(2):
            try:
                answer = self._chat(system, data, Selection.model_json_schema(), deadline)
                result = Selection.model_validate(answer)
                if len(result.selected_ids) > limit or any(
                    i not in allowed for i in result.selected_ids
                ):
                    raise ValueError("selection outside allowlist or limit")
                return result
            except (ValidationError, ValueError) as exc:
                if attempt:
                    raise LLMUnavailable("Invalid local LLM selection") from exc
        raise LLMUnavailable("Invalid local LLM selection")
