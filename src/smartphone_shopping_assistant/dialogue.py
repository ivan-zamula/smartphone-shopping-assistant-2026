"""Session-local dialogue over the read-only catalogue search pipeline."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from .contracts import FilterPatch, ProductCard, SearchFilters, SearchRequest, SearchResponse
from .llm import LocalLLM, literal_brands
from .search import SearchEngine


@dataclass
class SessionState:
    filters: SearchFilters = field(default_factory=SearchFilters)
    semantic_query: str = ""
    last_successful_ids: list[UUID] = field(default_factory=list)


class DialogueResponse(SearchResponse):
    comparison: list[ProductCard] = field(default_factory=list)


_UNSUPPORTED = re.compile(
    r"камер\w*|батаре\w*|аккумулятор\w*|процессор\w*|экран\w*|диспле\w*|"
    r"camera|battery|cpu|processor|display|screen|water\s*resistan\w*|waterproof\w*|"
    r"водозащит\w*|водонепроница\w*|конверт\w*|пересч\w*|convert",
    re.I,
)
_CHEAPER = re.compile(r"дешевле|подешевле|cheaper|less expensive", re.I)
_PURE_CHEAPER = re.compile(
    r"(?:покажи\s+)?(?:дешевле|подешевле)(?:\s+первого)?|"
    r"(?:show\s+)?(?:cheaper|less expensive)(?:\s+than\s+the\s+first)?",
    re.I,
)


def _merge(base: SearchFilters, patch: FilterPatch) -> SearchFilters:
    data = base.model_dump()
    for key in patch.model_fields_set:
        data[key] = getattr(patch, key)
    return SearchFilters.model_validate(data)


class SearchService:
    def __init__(self, engine: SearchEngine | None = None, llm: LocalLLM | None = None):
        self.engine = engine or SearchEngine()
        self.llm = llm or LocalLLM()

    def handle(
        self, text: str, session: SessionState, ui_changes: FilterPatch | None = None
    ) -> DialogueResponse:
        """Return a fresh response; only successful searches update positional references."""
        request_id = uuid4()
        deadline = time.monotonic() + 120
        filters = session.filters

        def check_deadline():
            if time.monotonic() >= deadline:
                raise TimeoutError

        def response(status, message, *, items=None, comparison=None, applied=None):
            return DialogueResponse(
                request_id=request_id,
                status=status,
                filters=applied or filters,
                items=items or [],
                comparison=comparison or [],
                message=message,
            )

        try:
            if not isinstance(text, str) or len(text) > 500:
                return response("needs_clarification", "Уточните запрос (не более 500 символов).")
            text = text.strip()
            if _UNSUPPORTED.search(text):
                return response(
                    "needs_clarification", "Эта характеристика или конвертация валюты недоступна."
                )
            if not text and not ui_changes:
                return response("needs_clarification", "Введите запрос или измените фильтры.")
            if text:
                known_brands = self.engine.known_brands()
                parsed = self.llm.parse(
                    text,
                    {
                        "filters": filters.model_dump(mode="json"),
                        "previous_result_count": len(session.last_successful_ids),
                        "known_brands": known_brands[:100],
                    },
                    deadline=deadline,
                )
                check_deadline()
                pure_cheaper = bool(_PURE_CHEAPER.fullmatch(text))
                if pure_cheaper and session.last_successful_ids:
                    parsed = parsed.model_copy(
                        update={"intent": "refine", "filter_changes": FilterPatch()}
                    )
                if parsed.intent == "clarify":
                    return response("needs_clarification", "Уточните условия поиска.")
                if parsed.intent == "compare":
                    positions = parsed.previous_result_positions
                    if len(positions) != 2 or any(
                        p > len(session.last_successful_ids) for p in positions
                    ):
                        return response(
                            "needs_clarification", "Укажите две позиции последней выдачи."
                        )
                    ids = [session.last_successful_ids[p - 1] for p in positions]
                    products = self.engine.get_products(ids)
                    check_deadline()
                    if any(i not in products for i in ids):
                        return response("error", "Один из товаров больше недоступен.")
                    cards = [products[i] for i in ids]
                    return response("ok", "Сравнение по текущим данным каталога.", comparison=cards)
                text_patch = parsed.filter_changes
                explicit, ambiguous = literal_brands(text, known_brands)
                if ambiguous:
                    return response("needs_clarification", "Уточните условие по бренду.")
                if explicit:
                    proposed = (
                        text_patch.brands if "brands" in text_patch.model_fields_set else None
                    )
                    if proposed is not None and {b.casefold() for b in proposed} != {
                        b.casefold() for b in explicit
                    }:
                        return response("needs_clarification", "Уточните условие по бренду.")
                    text_patch = text_patch.model_copy(update={"brands": explicit})
            else:
                parsed = None
                text_patch = FilterPatch()
            ui_changes = ui_changes or FilterPatch()
            for key in text_patch.model_fields_set & ui_changes.model_fields_set:
                if getattr(text_patch, key) != getattr(ui_changes, key):
                    return response(
                        "needs_clarification", f"Уточните противоречивые условия: {key}."
                    )
            merged = _merge(_merge(filters, text_patch), ui_changes)
            if _CHEAPER.search(text):
                if not session.last_successful_ids:
                    return response(
                        "needs_clarification", "Сначала выполните поиск для сравнения цены."
                    )
                first_id = session.last_successful_ids[0]
                first = self.engine.get_products([first_id]).get(first_id)
                check_deadline()
                if first is None:
                    return response("error", "Предыдущий товар больше недоступен.")
                # A new explicit price condition is ambiguous alongside a relative bound.
                if "price" in text_patch.model_fields_set | ui_changes.model_fields_set:
                    return response("needs_clarification", "Уточните условие цены.")
                old_price = merged.price
                if old_price and old_price.minimum is not None and old_price.minimum >= first.price:
                    return response("empty", "Нет товаров дешевле при текущем минимуме цены.")
                if old_price and old_price.maximum is not None and old_price.maximum < first.price:
                    ceiling = old_price.maximum
                    inclusive = old_price.include_maximum
                else:
                    ceiling = first.price
                    inclusive = False
                merged = _merge(
                    merged,
                    FilterPatch(
                        price={
                            "minimum": old_price.minimum if old_price else None,
                            "include_minimum": old_price.include_minimum if old_price else True,
                            "maximum": ceiling,
                            "include_maximum": inclusive,
                        }
                    ),
                )
            if text and not _CHEAPER.search(text):
                query = text
            else:
                query = session.semantic_query or "смартфон"
            request = SearchRequest(query=query, filters=merged)
            if time.monotonic() >= deadline:
                raise TimeoutError
            candidates = self.engine.retrieve(request, candidate_limit=20)
            check_deadline()
            if not candidates:
                session.filters, session.semantic_query = merged, query
                return response("empty", "По заданным условиям товары не найдены.", applied=merged)
            selected = self.llm.select(query, candidates[:10], limit=5, deadline=deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError
            if selected.abstain:
                session.filters, session.semantic_query = merged, query
                return response("empty", "Подходящие товары не выбраны.", applied=merged)
            allowed = {c.product.id for c in candidates[:10]}
            ids = selected.selected_ids
            if len(ids) > 5 or len(ids) != len(set(ids)) or any(i not in allowed for i in ids):
                raise ValueError("Invalid selected IDs")
            products = self.engine.get_products(ids)
            check_deadline()
            if any(i not in products or not self.engine._matches(products[i], merged) for i in ids):
                raise ValueError("Selected product changed")
            cards = [products[i] for i in ids]
            session.filters, session.semantic_query = merged, query
            session.last_successful_ids = ids.copy()
            return response(
                "ok",
                "Товары из каталога; валюта источника не подтверждена, пересчёта нет.",
                items=cards,
                applied=merged,
            )
        except Exception:
            return response("error", "Поиск сейчас недоступен. Повторите запрос позже.")
