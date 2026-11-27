"""Request-local LangGraph workflow for the read-only catalogue."""

from __future__ import annotations

import re
from time import monotonic, perf_counter
from typing import Literal, TypedDict
from uuid import UUID, uuid4

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
from .llm import LocalLLM, literal_brands
from .search import SearchEngine

NodeName = Literal[
    "parse_validate", "clarify", "compare", "prepare_query", "retrieve", "select", "finalize"
]
Status = Literal["ok", "empty", "needs_clarification", "error"]
ERROR_MESSAGE = "Поиск сейчас недоступен. Повторите запрос позже."


class WorkflowStep(Contract):
    node: NodeName
    elapsed_ms: float = Field(ge=0)


class DialogueResponse(SearchResponse):
    comparison: list[ProductCard] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)


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
    candidates: list[SearchCandidate]
    status: Status
    message: str
    items: list[ProductCard]
    comparison: list[ProductCard]
    session_request: SearchRequest
    selected_ids: list[UUID]
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
        query = (
            text if text and not _CHEAPER.search(text) else state["previous_query"] or "смартфон"
        )
        return {"request": SearchRequest(query=query, filters=state["merged_filters"])}

    def retrieve(state: WorkflowState) -> WorkflowState:
        request = state["request"]
        _check_deadline(state)
        candidates = engine.retrieve(request, candidate_limit=20)
        _check_deadline(state)
        if not candidates:
            return {
                "status": "empty",
                "message": "По заданным условиям товары не найдены.",
                "session_request": request,
            }
        return {"candidates": candidates}

    def after_retrieve(state: WorkflowState) -> Literal["select", "finalize"]:
        return "select" if state.get("candidates") else "finalize"

    def select(state: WorkflowState) -> WorkflowState:
        request = state["request"]
        candidates = state["candidates"][:10]
        selection = llm.select(request.query, candidates, limit=5, deadline=state["deadline"])
        _check_deadline(state)
        if selection.abstain:
            return {
                "status": "empty",
                "message": "Подходящие товары не выбраны.",
                "session_request": request,
            }
        allowed = {candidate.product.id for candidate in candidates}
        ids = selection.selected_ids
        if len(ids) > 5 or len(ids) != len(set(ids)) or any(i not in allowed for i in ids):
            raise ValueError("Invalid selected IDs")
        products = engine.get_products(ids)
        _check_deadline(state)
        if any(i not in products or not engine._matches(products[i], request.filters) for i in ids):
            raise ValueError("Selected product changed")
        return {
            "status": "ok",
            "message": "Товары из каталога; валюта источника не подтверждена, пересчёта нет.",
            "items": [products[i] for i in ids],
            "session_request": request,
            "selected_ids": ids,
        }

    def finalize(state: WorkflowState) -> WorkflowState:
        _check_deadline(state)
        request = state.get("session_request")
        return {
            "response": DialogueResponse(
                request_id=state["request_id"],
                status=state["status"],
                filters=request.filters if request is not None else state["filters"],
                message=state["message"],
                items=state.get("items", []),
                comparison=state.get("comparison", []),
            )
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("parse_validate", parse_validate)
    graph.add_node("clarify", clarify)
    graph.add_node("compare", compare)
    graph.add_node("prepare_query", prepare_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("select", select)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "parse_validate")
    graph.add_conditional_edges("parse_validate", after_parse)
    graph.add_edge("clarify", "finalize")
    graph.add_edge("compare", "finalize")
    graph.add_edge("prepare_query", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve)
    graph.add_edge("select", "finalize")
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
    except Exception:
        return (
            DialogueResponse(
                request_id=initial["request_id"],
                status="error",
                filters=filters,
                message=ERROR_MESSAGE,
                steps=steps,
            ),
            None,
            None,
        )
    response.steps = steps
    return response, state.get("session_request"), state.get("selected_ids")
