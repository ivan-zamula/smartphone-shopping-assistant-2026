"""Request-local LangGraph workflow for the read-only catalogue."""

from __future__ import annotations

import re
from itertools import zip_longest
from time import monotonic, perf_counter
from typing import Literal, TypedDict
from uuid import UUID, uuid4

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context
from pydantic import Field

from .contracts import (
    Contract,
    FilterPatch,
    ParsedIntent,
    ProductCard,
    SearchCandidate,
    SearchFilters,
    SearchRequest,
    SearchResponse,
)
from .llm import (
    InvalidModelResponse,
    LLMCallLimitExceeded,
    LLMDeadlineExceeded,
    LLMUnavailable,
    LocalLLM,
    OllamaCalls,
    ResultEvaluation,
    literal_brands,
    ollama_calls,
)
from .search import SearchEngine

NodeName = Literal[
    "parse_validate",
    "clarify",
    "compare",
    "prepare_query",
    "retrieve",
    "select",
    "evaluate",
    "rewrite",
    "finalize",
]
Status = Literal["ok", "empty", "needs_clarification", "error"]
ERROR_MESSAGE = "Поиск сейчас недоступен. Повторите запрос позже."


StopReason = Literal[
    "sufficient",
    "retry_complete",
    "rewrite_rejected",
    "sql_empty",
    "clarification",
    "comparison",
    "empty",
    "search_error",
    "model_unavailable",
    "invalid_model",
    "deadline",
    "call_limit",
    "graph_limit",
]
STOP_MESSAGES = {
    "search_error": ERROR_MESSAGE,
    "model_unavailable": "Локальная модель недоступна. Повторите запрос позже.",
    "invalid_model": "Локальная модель вернула некорректный ответ.",
    "deadline": "Время поиска истекло. Повторите запрос.",
    "call_limit": "Достигнут лимит обращений к локальной модели.",
    "graph_limit": "Достигнут лимит шагов поиска.",
}


class NodeFailed(RuntimeError):
    def __init__(self, node: NodeName, *, entered: bool):
        self.node = node
        self.entered = entered


def _stop_reason(error: BaseException) -> StopReason:
    if isinstance(error, (TimeoutError, LLMDeadlineExceeded)):
        return "deadline"
    if isinstance(error, LLMCallLimitExceeded):
        return "call_limit"
    if isinstance(error, GraphRecursionError):
        return "graph_limit"
    if isinstance(error, InvalidModelResponse):
        return "invalid_model"
    if isinstance(error, LLMUnavailable):
        return "model_unavailable"
    return "search_error"


class WorkflowStep(Contract):
    node: NodeName
    elapsed_ms: float = Field(ge=0)
    failed: bool = False


class WorkflowDiagnostics(Contract):
    queries: list[str] = Field(max_length=2)
    search_attempts: int = Field(ge=0, le=2)
    ollama_calls: int = Field(ge=0, le=8)
    candidate_count: int = Field(ge=0, le=40)
    stop_reason: StopReason


class DialogueResponse(SearchResponse):
    comparison: list[ProductCard] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
    diagnostics: WorkflowDiagnostics | None = None


class WorkflowState(TypedDict, total=False):
    request_id: UUID
    deadline: float
    text: str
    ui_changes: FilterPatch
    filters: SearchFilters
    previous_query: str
    previous_ids: list[UUID]
    parsed: ParsedIntent
    merged_filters: SearchFilters
    clarification: str
    request: SearchRequest
    original_query: str
    queries: list[str]
    first_candidates: list[SearchCandidate]
    candidates: list[SearchCandidate]
    evaluation: ResultEvaluation
    stop_reason: StopReason
    status: Status
    message: str
    items: list[ProductCard]
    comparison: list[ProductCard]
    session_request: SearchRequest | None
    selected_ids: list[UUID] | None
    response: DialogueResponse


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


def _check_deadline(state: WorkflowState) -> None:
    if monotonic() >= state["deadline"]:
        raise TimeoutError


def _query_key(query: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", query.casefold()))


def _anchored_query(original: str, proposal: str) -> str | None:
    """Keep the original wording as a retrieval anchor for model-generated terms."""
    proposal = proposal.strip()
    if not 1 <= len(proposal) <= 500 or not _query_key(proposal) or _UNSUPPORTED.search(proposal):
        return None
    original_tokens = set(_query_key(original))
    if any(
        any(char.isdecimal() for char in word) and word not in original_tokens
        for word in _query_key(proposal)
    ):
        return None
    numbers = r"\d+(?:[.,]\d+)*"
    if set(re.findall(numbers, proposal)) - set(re.findall(numbers, original)):
        return None
    if _query_key(original) == _query_key(proposal):
        return original
    if not re.match(re.escape(original) + r"(?!\w)", proposal, re.I):
        proposal = f"{original}. {proposal}"
    return proposal if len(proposal) <= 500 else None


def _merge_candidates(
    first: list[SearchCandidate], second: list[SearchCandidate]
) -> list[SearchCandidate]:
    merged: dict[UUID, SearchCandidate] = {}
    for pair in zip_longest(first[:20], second[:20]):
        for candidate in pair:
            if candidate is not None:
                merged.setdefault(candidate.product.id, candidate)
    return list(merged.values())


def build_workflow(engine: SearchEngine, llm: LocalLLM) -> CompiledStateGraph:
    def parse_validate(state: WorkflowState) -> WorkflowState:
        text = state["text"]
        ui_changes = state["ui_changes"]
        filters = state["filters"]
        previous_ids = state["previous_ids"]
        if not isinstance(text, str) or len(text) > 500:
            return {"clarification": "Уточните запрос (не более 500 символов)."}
        text = text.strip()
        if _UNSUPPORTED.search(text):
            return {"clarification": "Эта характеристика или конвертация валюты недоступна."}
        if not text and not ui_changes:
            return {"clarification": "Введите запрос или измените фильтры."}
        update: WorkflowState = {"text": text}
        text_patch = FilterPatch()
        if text:
            known_brands = engine.known_brands()
            _check_deadline(state)
            parsed = llm.parse(
                text,
                {
                    "filters": filters.model_dump(mode="json"),
                    "previous_result_count": len(previous_ids),
                    "known_brands": known_brands[:100],
                },
                deadline=state["deadline"],
            )
            _check_deadline(state)
            if _PURE_CHEAPER.fullmatch(text) and previous_ids:
                parsed = parsed.model_copy(
                    update={"intent": "refine", "filter_changes": FilterPatch()}
                )
            update["parsed"] = parsed
            if parsed.intent == "clarify":
                return {**update, "clarification": "Уточните условия поиска."}
            if parsed.intent == "compare":
                positions = parsed.previous_result_positions
                if len(positions) != 2 or any(p > len(previous_ids) for p in positions):
                    return {**update, "clarification": "Укажите две позиции последней выдачи."}
                return update
            text_patch = parsed.filter_changes
            explicit, ambiguous = literal_brands(text, known_brands)
            if ambiguous:
                return {**update, "clarification": "Уточните условие по бренду."}
            if explicit:
                proposed = text_patch.brands
                if proposed is not None and {b.casefold() for b in proposed} != {
                    b.casefold() for b in explicit
                }:
                    return {**update, "clarification": "Уточните условие по бренду."}
                text_patch = text_patch.model_copy(update={"brands": explicit})
        for key in text_patch.model_fields_set & ui_changes.model_fields_set:
            if getattr(text_patch, key) != getattr(ui_changes, key):
                return {**update, "clarification": f"Уточните противоречивые условия: {key}."}
        merged = _merge(_merge(filters, text_patch), ui_changes)
        if _CHEAPER.search(text):
            if not previous_ids:
                return {**update, "clarification": "Сначала выполните поиск для сравнения цены."}
            first_id = previous_ids[0]
            first = engine.get_products([first_id]).get(first_id)
            _check_deadline(state)
            if first is None:
                return {"status": "error", "message": "Предыдущий товар больше недоступен."}
            if "price" in text_patch.model_fields_set | ui_changes.model_fields_set:
                return {**update, "clarification": "Уточните условие цены."}
            old_price = merged.price
            if old_price and old_price.minimum is not None and old_price.minimum >= first.price:
                return {
                    "status": "empty",
                    "message": "Нет товаров дешевле при текущем минимуме цены.",
                }
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
        return {**update, "merged_filters": merged}

    def after_parse(
        state: WorkflowState,
    ) -> Literal["clarify", "compare", "prepare_query", "finalize"]:
        if "clarification" in state:
            return "clarify"
        if "status" in state:
            return "finalize"
        parsed = state.get("parsed")
        if parsed and parsed.intent == "compare":
            return "compare"
        return "prepare_query"

    def clarify(state: WorkflowState) -> WorkflowState:
        return {"status": "needs_clarification", "message": state["clarification"]}

    def compare(state: WorkflowState) -> WorkflowState:
        positions = state["parsed"].previous_result_positions
        ids = [state["previous_ids"][p - 1] for p in positions]
        products = engine.get_products(ids)
        _check_deadline(state)
        if any(i not in products for i in ids):
            return {"status": "error", "message": "Один из товаров больше недоступен."}
        return {
            "status": "ok",
            "message": "Сравнение по текущим данным каталога.",
            "comparison": [products[i] for i in ids],
        }

    def prepare_query(state: WorkflowState) -> WorkflowState:
        text = state["text"]
        original = (
            text if text and not _CHEAPER.search(text) else state["previous_query"] or "смартфон"
        )
        filters = state["merged_filters"]
        proposed = llm.prepare_query(
            original, filters.model_dump(mode="json"), deadline=state["deadline"]
        )
        _check_deadline(state)
        query = _anchored_query(original, proposed) or original
        return {"original_query": original, "request": SearchRequest(query=query, filters=filters)}

    def retrieve(state: WorkflowState) -> WorkflowState:
        request = state["request"]
        queries = [*state.get("queries", []), request.query]
        result = engine.retrieve_with_status(
            request, candidate_limit=20, deadline=state["deadline"]
        )
        _check_deadline(state)
        if result.sql_empty:
            return {
                "queries": queries,
                "candidates": [],
                "selected_ids": [],
                "status": "empty",
                "stop_reason": "sql_empty",
                "message": "По заданным условиям товары не найдены.",
            }
        candidates = result.candidates[:20]
        first = state.get("first_candidates", candidates)
        pool = (
            _merge_candidates(first, candidates)
            if len(queries) == 2
            else _merge_candidates(first, [])
        )
        return {"queries": queries, "first_candidates": first, "candidates": pool}

    def after_retrieve(state: WorkflowState) -> Literal["select", "finalize"]:
        return "finalize" if state.get("stop_reason") == "sql_empty" else "select"

    def select(state: WorkflowState) -> WorkflowState:
        candidates = state["candidates"][:10]
        ids = []
        if candidates:
            selection = llm.select(
                state["original_query"], candidates, limit=5, deadline=state["deadline"]
            )
            _check_deadline(state)
            allowed = {candidate.product.id for candidate in candidates}
            ids = selection.selected_ids
            if (
                selection.abstain != (not ids)
                or len(ids) > 5
                or len(ids) != len(set(ids))
                or any(i not in allowed for i in ids)
            ):
                raise InvalidModelResponse("Invalid selected IDs")
        if not ids:
            return {
                "status": "empty",
                "message": "Подходящие товары не выбраны.",
                "selected_ids": [],
            }
        return {
            "status": "ok",
            "message": "Товары из каталога; валюта источника не подтверждена, пересчёта нет.",
            "selected_ids": ids,
        }

    def after_select(state: WorkflowState) -> Literal["evaluate", "finalize"]:
        return "evaluate" if len(state["queries"]) == 1 else "finalize"

    def evaluate(state: WorkflowState) -> WorkflowState:
        request = state["request"]
        by_id = {candidate.product.id: candidate for candidate in state["candidates"]}
        selected = [by_id[i] for i in state["selected_ids"]]
        evaluation = llm.evaluate(
            state["original_query"],
            request.query,
            request.filters.model_dump(mode="json"),
            selected,
            deadline=state["deadline"],
        )
        _check_deadline(state)
        if (evaluation.reason == "no_candidates") != (not selected):
            raise InvalidModelResponse("Evaluation does not match selected products")
        return {"evaluation": evaluation}

    def after_evaluate(state: WorkflowState) -> Literal["finalize", "rewrite"]:
        return "finalize" if state["evaluation"].sufficient else "rewrite"

    def rewrite(state: WorkflowState) -> WorkflowState:
        proposed = state["evaluation"].rewrite
        original = state["original_query"]
        keys = {_query_key(q) for q in [original, *state["queries"]]}
        query = _anchored_query(original, proposed) if proposed is not None else None
        if query is None or _query_key(proposed) in keys or _query_key(query) in keys:
            return {"stop_reason": "rewrite_rejected"}
        request = state["request"]
        return {"request": SearchRequest(query=query, filters=request.filters)}

    def after_rewrite(state: WorkflowState) -> Literal["retrieve", "finalize"]:
        return "finalize" if state.get("stop_reason") == "rewrite_rejected" else "retrieve"

    def finalize(state: WorkflowState) -> WorkflowState:
        request = state.get("request")
        ids = (state.get("selected_ids") or []) if state["status"] == "ok" else []
        items = []
        if ids and request is not None:
            products = engine.get_products(ids)
            _check_deadline(state)
            if any(
                i not in products
                or products[i].id != i
                or not engine._matches(products[i], request.filters)
                for i in ids
            ):
                raise ValueError("Selected product changed")
            items = [products[i] for i in ids]
        session_request = (
            SearchRequest(query=state["original_query"], filters=request.filters)
            if request is not None and state["status"] in ("ok", "empty")
            else None
        )
        return {
            "session_request": session_request,
            "selected_ids": ids if items else None,
            "response": DialogueResponse(
                request_id=state["request_id"],
                status=state["status"],
                filters=session_request.filters
                if session_request is not None
                else state["filters"],
                message=state["message"],
                items=items,
                comparison=state.get("comparison", []),
            ),
        }

    def checked_node(name: NodeName, node):
        def run(state: WorkflowState) -> WorkflowState:
            entered = False
            try:
                _check_deadline(state)
                entered = True
                return node(state)
            except Exception as exc:
                raise NodeFailed(name, entered=entered) from exc

        return run

    graph = StateGraph(WorkflowState)
    for name, node in (
        ("parse_validate", parse_validate),
        ("clarify", clarify),
        ("compare", compare),
        ("prepare_query", prepare_query),
        ("retrieve", retrieve),
        ("select", select),
        ("evaluate", evaluate),
        ("rewrite", rewrite),
        ("finalize", finalize),
    ):
        graph.add_node(name, checked_node(name, node))
    graph.add_edge(START, "parse_validate")
    graph.add_conditional_edges("parse_validate", after_parse)
    graph.add_edge("clarify", "finalize")
    graph.add_edge("compare", "finalize")
    graph.add_edge("prepare_query", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve)
    graph.add_conditional_edges("select", after_select)
    graph.add_conditional_edges("evaluate", after_evaluate)
    graph.add_conditional_edges("rewrite", after_rewrite)
    graph.add_edge("finalize", END)
    return graph.compile()


def run_workflow(
    graph: CompiledStateGraph,
    text: str,
    ui_changes: FilterPatch | None,
    filters: SearchFilters,
    previous_query: str,
    previous_ids: list[UUID],
) -> tuple[DialogueResponse, SearchRequest | None, list[UUID] | None]:
    initial: WorkflowState = {
        "request_id": uuid4(),
        "deadline": monotonic() + 180,
        "text": text,
        "ui_changes": ui_changes or FilterPatch(),
        "filters": filters,
        "previous_query": previous_query,
        "previous_ids": previous_ids.copy(),
    }
    steps: list[WorkflowStep] = []
    state = initial.copy()
    started = perf_counter()
    calls = OllamaCalls()
    token = ollama_calls.set(calls)
    failed_reason: StopReason | None = None
    retrieval_failed = False
    try:
        with tracing_context(enabled=False):
            for event in graph.stream(
                initial, config={"recursion_limit": 12}, stream_mode="updates"
            ):
                for node, update in event.items():
                    now = perf_counter()
                    steps.append(WorkflowStep(node=node, elapsed_ms=(now - started) * 1000))
                    started = now
                    state.update(update)
        response = state["response"]
        _check_deadline(state)
    except Exception as exc:
        error = exc
        if isinstance(exc, NodeFailed):
            retrieval_failed = exc.node == "retrieve" and exc.entered
            steps.append(
                WorkflowStep(
                    node=exc.node, elapsed_ms=(perf_counter() - started) * 1000, failed=True
                )
            )
            error = exc.__cause__ or exc
        failed_reason = _stop_reason(error)
        response = DialogueResponse(
            request_id=initial["request_id"],
            status="error",
            filters=filters,
            message=STOP_MESSAGES[failed_reason],
        )
    finally:
        ollama_calls.reset(token)

    queries = state.get("queries", [])
    if retrieval_failed:
        queries = [*queries, state["request"].query]
    reason = failed_reason or state.get("stop_reason")
    if reason is None:
        if len(queries) == 2:
            reason = "retry_complete"
        elif "evaluation" in state:
            reason = "sufficient"
        elif response.status == "needs_clarification":
            reason = "clarification"
        elif response.comparison:
            reason = "comparison"
        else:
            reason = "empty" if response.status == "empty" else "search_error"
    response.steps = steps
    response.diagnostics = WorkflowDiagnostics(
        queries=queries,
        search_attempts=len(queries),
        ollama_calls=calls.count,
        candidate_count=len(state.get("candidates", [])),
        stop_reason=reason,
    )
    if failed_reason:
        return response, None, None
    return response, state.get("session_request"), state.get("selected_ids")
