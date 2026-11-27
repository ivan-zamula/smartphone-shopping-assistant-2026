"""Synthetic, isolated dialogue state and comparison checks."""

from decimal import Decimal
from uuid import UUID

import pytest

from smartphone_shopping_assistant.contracts import (
    FilterPatch,
    NumericRange,
    ParsedIntent,
    ProductCard,
    SearchCandidate,
)
from smartphone_shopping_assistant.dialogue import SearchService, SessionState
from smartphone_shopping_assistant.llm import ResultEvaluation, Selection
from smartphone_shopping_assistant.search import Retrieval, SearchEngine


def card(number, price=100):
    return ProductCard(
        id=UUID(int=number),
        name=f"Synthetic {number}",
        brand="Test",
        model=str(number),
        ram_gb=8,
        storage_gb=128,
        color="black",
        carrier_unlocked=True,
        price=Decimal(price),
        currency="USD",
        currency_status="unconfirmed",
        source_currency=None,
        conversion_applied=False,
        raw_fields={},
        text="synthetic",
    )


class FakeEngine:
    _matches = staticmethod(SearchEngine._matches)

    def __init__(self):
        self.products = {c.id: c for c in [card(1, 100), card(2, 80)]}
        self.requests = []
        self.fail = False

    def retrieve_with_status(self, request, candidate_limit, *, deadline):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("database down")
        candidates = [
            SearchCandidate(product=c, vector_score=0.8)
            for c in self.products.values()
            if self._matches(c, request.filters)
        ]
        return Retrieval(candidates[:candidate_limit], sql_empty=not candidates)

    def get_products(self, ids):
        return {i: self.products[i] for i in ids if i in self.products}

    def known_brands(self):
        return sorted({item.brand for item in self.products.values()})


def test_literal_brand_survives_missing_model_patch():
    service, engine, llm, session = setup()
    engine.products[UUID(int=1)] = card(1).model_copy(update={"brand": "Samsung"})
    result = service.handle("Samsung с 8 ГБ ОЗУ", session)
    assert result.status == "ok"
    assert result.filters.brands == ["Samsung"]
    assert all(item.brand == "Samsung" for item in result.items)
    llm.next_intent = ParsedIntent(intent="search", filter_changes=FilterPatch(brands=["Test"]))
    assert service.handle("Samsung с 8 ГБ ОЗУ", session).status == "needs_clarification"
    assert service.handle("не Samsung", session).status == "needs_clarification"


class FakeLLM:
    def __init__(self):
        self.next_intent = ParsedIntent(intent="search")
        self.select_calls = 0
        self.evaluations = []
        self.context_ids = []
        self.prepared = None
        self.rewrite = None

    def parse(self, query, contextual_info, deadline):
        return self.next_intent

    def prepare_query(self, query, filters, *, deadline):
        return query if self.prepared is None else self.prepared

    def evaluate(self, original, query, filters, selected, *, deadline):
        self.evaluations.append(selected)
        sufficient = bool(selected) and self.rewrite is None
        return ResultEvaluation(
            sufficient=sufficient,
            reason="relevant" if sufficient else "poor_match" if selected else "no_candidates",
            rewrite=self.rewrite,
        )

    def select(self, query, candidates, limit, deadline):
        self.select_calls += 1
        self.context_ids.append([c.product.id for c in candidates])
        ids = [c.product.id for c in candidates[:limit]]
        return Selection(selected_ids=ids, abstain=not ids, reason_codes=[])


def setup():
    engine, llm = FakeEngine(), FakeLLM()
    return SearchService(engine, llm), engine, llm, SessionState()


def test_unsupported_water_protection_requires_clarification():
    service, engine, llm, session = setup()
    for query in ("water resistance", "waterproof", "водозащита"):
        assert service.handle(query, session).status == "needs_clarification"
    assert not engine.requests
    assert llm.select_calls == 0


def test_search_cheaper_compare_and_current_rows():
    service, engine, llm, session = setup()
    first = service.handle("phone", session)
    assert first.status == "ok" and len(first.items) == 2
    assert [step.node for step in first.steps] == [
        "parse_validate",
        "prepare_query",
        "retrieve",
        "select",
        "evaluate",
        "finalize",
    ]
    assert all(step.elapsed_ms >= 0 for step in first.steps)
    assert len(engine.requests) == 1
    llm.next_intent = ParsedIntent(intent="refine")
    cheaper = service.handle("покажи дешевле", session)
    assert cheaper.status == "ok"
    assert cheaper.items[0].id == UUID(int=2)
    assert engine.requests[-1].filters.price == NumericRange(
        maximum=Decimal(100), include_maximum=False
    )
    llm.next_intent = ParsedIntent(intent="compare", previous_result_positions=[1, 2])
    clarification = service.handle("compare first two", session)
    assert clarification.status == "needs_clarification"
    assert [step.node for step in clarification.steps] == ["parse_validate", "clarify", "finalize"]
    llm.next_intent = ParsedIntent(intent="search")
    service.handle("phone", session, FilterPatch(price=None))
    llm.next_intent = ParsedIntent(intent="compare", previous_result_positions=[1, 2])
    before = llm.select_calls
    engine.products[UUID(int=1)] = card(1, 90)
    searches_before = len(engine.requests)
    comparison = service.handle("compare first two", session)
    assert [step.node for step in comparison.steps] == ["parse_validate", "compare", "finalize"]
    assert len(engine.requests) == searches_before
    assert [c.price for c in comparison.comparison] == [Decimal(90), Decimal(80)]
    assert llm.select_calls == before and comparison.items == []


def test_ui_patch_clear_conflict_and_isolation():
    service, engine, llm, session = setup()
    other = SessionState()
    assert service.handle("", session, FilterPatch(brands=["Test"])).status == "ok"
    assert session.filters.brands == ["Test"] and other.filters.brands is None
    llm.next_intent = ParsedIntent(intent="refine", filter_changes=FilterPatch(brands=["Other"]))
    conflict = service.handle("Other", session, FilterPatch(brands=["Test"]))
    assert conflict.status == "needs_clarification"
    assert session.filters.brands == ["Test"]
    llm.next_intent = ParsedIntent(intent="refine", filter_changes=FilterPatch(brands=None))
    assert service.handle("clear brands", session).status == "ok"
    assert session.filters.brands is None


def test_failure_empty_and_unknown_comparison_ids():
    service, engine, llm, session = setup()
    assert service.handle("phone", session).status == "ok"
    saved = session.last_successful_ids.copy()
    engine.fail = True
    assert service.handle("phone", session).items == []
    assert session.last_successful_ids == saved
    engine.fail = False
    llm.next_intent = ParsedIntent(intent="compare", previous_result_positions=[1, 2])
    del engine.products[UUID(int=2)]
    assert service.handle("compare first two", session).status == "error"
    llm.next_intent = ParsedIntent(intent="search", filter_changes=FilterPatch(brands=["Nope"]))
    assert service.handle("Nope", session).status == "empty"
    assert session.last_successful_ids == saved


def test_cheaper_keeps_existing_price_minimum():
    service, engine, llm, session = setup()
    assert service.handle("", session, FilterPatch(price={"minimum": 50})).status == "ok"
    llm.next_intent = ParsedIntent(intent="refine")
    result = service.handle("покажи дешевле", session)
    assert result.status == "ok"
    assert engine.requests[-1].filters.price.minimum == Decimal(50)
    assert engine.requests[-1].filters.price.maximum == Decimal(100)
    assert engine.requests[-1].filters.price.include_maximum is False


def test_pure_cheaper_ignores_model_intent_and_patch():
    for intent in ("clarify", "compare"):
        service, engine, llm, session = setup()
        assert service.handle("", session, FilterPatch(price={"minimum": 50})).status == "ok"
        llm.next_intent = ParsedIntent(intent=intent, filter_changes=FilterPatch(brands=["Other"]))
        result = service.handle("Покажи дешевле первого", session)
        assert result.status == "ok"
        assert result.filters.price == NumericRange(
            minimum=Decimal(50), maximum=Decimal(100), include_maximum=False
        )
        assert result.filters.brands is None
    service, engine, llm, session = setup()
    llm.next_intent = ParsedIntent(intent="clarify")
    assert service.handle("cheaper", session).status == "needs_clarification"
    session.last_successful_ids = [UUID(int=1)]
    assert service.handle("cheaper with 16 GB RAM", session).status == "needs_clarification"


@pytest.mark.parametrize("compare", [False, True])
def test_timeout_after_hydration(monkeypatch, compare):
    service, engine, llm, session = setup()
    saved = [UUID(int=1), UUID(int=2)]
    session.last_successful_ids = saved.copy()
    if compare:
        llm.next_intent = ParsedIntent(intent="compare", previous_result_positions=[1, 2])
    now = 0

    def hydrate(ids):
        nonlocal now
        now = 181
        return {i: engine.products[i] for i in ids}

    monkeypatch.setattr(engine, "get_products", hydrate)
    monkeypatch.setattr("smartphone_shopping_assistant.workflow.monotonic", lambda: now)
    result = service.handle("compare" if compare else "phone", session)
    assert result.status == "error" and not result.items and not result.comparison
    assert result.diagnostics.stop_reason == "deadline"
    assert result.steps[-1].node == ("compare" if compare else "finalize")
    assert result.steps[-1].failed
    assert session.last_successful_ids == saved and session.semantic_query == ""


@pytest.mark.parametrize("empty_vectors", [False, True])
def test_retry_interleaves_ranks_and_preserves_original_and_filters(monkeypatch, empty_vectors):
    service, engine, llm, session = setup()
    llm.rewrite = "business handset"
    llm.prepared = "work phone"
    engine.products.update({UUID(int=i): card(i) for i in range(3, 40)})
    first = range(1, 21) if not empty_vectors else []
    second = [1, *range(21, 40)]
    orders = iter([first, second])

    def retrieve(request, candidate_limit, *, deadline):
        engine.requests.append(request)
        score = 0.1 if len(engine.requests) == 1 else 0.99
        return Retrieval(
            [
                SearchCandidate(product=engine.products[UUID(int=i)], vector_score=score)
                for i in next(orders)
            ]
        )

    monkeypatch.setattr(engine, "retrieve_with_status", retrieve)
    original = "work phone 128 GB"
    result = service.handle(original, session, FilterPatch(brands=["Test"]))
    assert result.status == "ok"
    assert result.diagnostics.stop_reason == "retry_complete"
    assert result.diagnostics.queries == [
        original + ". work phone",
        original + ". business handset",
    ]
    assert result.diagnostics.search_attempts == 2
    assert result.diagnostics.candidate_count == (20 if empty_vectors else 39)
    assert all(request.filters.brands == ["Test"] for request in engine.requests)
    assert session.semantic_query == original
    assert len(llm.evaluations) == 1
    context = llm.context_ids[-1]
    expected = second[:10] if empty_vectors else [1, 2, 21, 3, 22, 4, 23, 5, 24, 6]
    assert [i.int for i in context] == expected
    assert [p.id for p in result.items] == context[:5]


@pytest.mark.parametrize("rewrite", [" ", "PHONE!", "x" * 501, "phone S25"])
def test_unusable_rewrite_finishes_without_retry(rewrite):
    service, engine, llm, session = setup()
    llm.rewrite = rewrite
    result = service.handle("phone", session)
    assert result.status == "ok"
    assert result.diagnostics.stop_reason == "rewrite_rejected"
    assert len(engine.requests) == 1


def test_sql_empty_and_failed_retrieval_skip_evaluation_and_retry():
    service, engine, llm, session = setup()
    llm.rewrite = "handset"
    result = service.handle("phone", session, FilterPatch(brands=["Missing"]))
    assert result.status == "empty"
    assert result.diagnostics.stop_reason == "sql_empty"
    assert result.diagnostics.search_attempts == 1
    engine.fail = True
    result = service.handle("handset", session, FilterPatch(brands=None))
    assert result.status == "error"
    assert result.diagnostics.stop_reason == "search_error"
    assert result.diagnostics.search_attempts == 1
    assert result.steps[-1].failed
    assert not llm.evaluations
    assert session.filters.brands == ["Missing"]
    assert session.semantic_query == "phone"


@pytest.mark.parametrize("exhausted", [False, True])
def test_physical_http_budget_includes_repairs_and_resets(exhausted):
    import json

    import httpx

    from smartphone_shopping_assistant.llm import LocalLLM, ollama_calls

    engine = FakeEngine()
    intent = {"intent": "search"}
    prepared = {"query": "phone"}
    selection = {"selected_positions": [1], "reason_codes": []}
    evaluation = {"sufficient": False, "reason": "poor_match", "rewrite": "handset"}
    enough = {"sufficient": True, "reason": "relevant", "rewrite": None}

    # Force all applicable repair paths through actual HTTP requests.
    replies = iter(
        [
            None,
            intent,
            prepared,
            None,
            selection,
            *([None] if exhausted else []),
            evaluation,
            None,
            *([] if exhausted else [selection]),
            intent,
            prepared,
            selection,
            enough,
        ]
    )
    requests = []

    def handler(request):
        requests.append(request)
        reply = next(replies)
        content = json.dumps(reply) if reply else "{"
        return httpx.Response(200, json={"message": {"content": content}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = SearchService(engine, LocalLLM(client=client))
        first = SessionState(semantic_query="prior", last_successful_ids=[UUID(int=2)])
        other = SessionState()
        result = service.handle("phone", first)
        assert len(requests) == result.diagnostics.ollama_calls == 8
        assert result.diagnostics.search_attempts == 2
        assert ollama_calls.get() is None
        if exhausted:
            assert result.status == "error"
            assert result.diagnostics.stop_reason == "call_limit"
            assert result.steps[-1].node == "select" and result.steps[-1].failed
            assert not result.items
            assert first.semantic_query == "prior"
            assert first.last_successful_ids == [UUID(int=2)]
        else:
            assert result.status == "ok"
        assert other.semantic_query == "" and not other.last_successful_ids
        next_result = service.handle("phone", other)
        assert next_result.status == "ok"
        assert next_result.diagnostics.ollama_calls == 4
        assert len(requests) == 12
        assert ollama_calls.get() is None


def test_retry_failure_does_not_return_first_selection(monkeypatch):
    service, engine, llm, session = setup()
    old = [UUID(int=2)]
    session.last_successful_ids = old.copy()
    llm.rewrite = "handset"
    retrieve = engine.retrieve_with_status

    def fail_on_retry(*args, **kwargs):
        if engine.requests:
            raise RuntimeError("Qdrant down")
        return retrieve(*args, **kwargs)

    monkeypatch.setattr(engine, "retrieve_with_status", fail_on_retry)
    response = service.handle("phone", session)
    assert response.status == "error" and not response.items
    assert response.diagnostics.stop_reason == "search_error"
    assert response.diagnostics.search_attempts == 2
    assert response.steps[-1].node == "retrieve" and response.steps[-1].failed
    assert session.last_successful_ids == old


def test_retry_rehydrates_and_rechecks_exact_filters(monkeypatch):
    service, engine, llm, session = setup()
    llm.rewrite = "handset"
    hydrate = engine.get_products

    def changed_products(ids):
        assert len(engine.requests) == 2
        products = hydrate(ids)
        products[ids[0]] = products[ids[0]].model_copy(update={"brand": "Changed"})
        return products

    monkeypatch.setattr(engine, "get_products", changed_products)
    response = service.handle("phone", session, FilterPatch(brands=["Test"]))
    assert response.status == "error" and not response.items
    assert response.diagnostics.stop_reason == "search_error"
    assert response.steps[-1].node == "finalize" and response.steps[-1].failed
    assert session.filters.brands is None


@pytest.mark.parametrize("failure", ["invalid_model", "model_unavailable"])
def test_model_failures_have_distinct_status_and_no_retry(failure, monkeypatch):
    from smartphone_shopping_assistant.llm import InvalidModelResponse, LLMUnavailable

    service, engine, llm, session = setup()

    def fail(*args, **kwargs):
        if failure == "invalid_model":
            raise InvalidModelResponse("invalid")
        raise LLMUnavailable("unavailable")

    monkeypatch.setattr(llm, "evaluate", fail)
    result = service.handle("phone", session)
    assert result.status == "error" and not result.items
    assert result.diagnostics.stop_reason == failure
    assert result.steps[-1].node == "evaluate" and result.steps[-1].failed
    assert len(engine.requests) == 1 and not session.last_successful_ids


def test_sql_empty_on_retry_does_not_commit_first_selection(monkeypatch):
    service, engine, llm, session = setup()
    saved = [UUID(int=42)]
    session.last_successful_ids = saved.copy()
    llm.rewrite = "handset"
    retrieve = engine.retrieve_with_status

    def sql_empty_on_retry(*args, **kwargs):
        if engine.requests:
            return Retrieval([], sql_empty=True)
        return retrieve(*args, **kwargs)

    monkeypatch.setattr(engine, "retrieve_with_status", sql_empty_on_retry)
    result = service.handle("phone", session)
    assert result.status == "empty" and not result.items
    assert result.diagnostics.stop_reason == "sql_empty"
    assert result.diagnostics.search_attempts == 2
    assert session.last_successful_ids == saved


def test_concurrent_requests_count_http_calls_independently():
    import json
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import httpx

    from smartphone_shopping_assistant.llm import LocalLLM

    barrier = Barrier(2)

    def handler(request):
        payload = json.loads(request.content)
        fields = payload["format"]["properties"]
        if "intent" in fields:
            barrier.wait(timeout=5)
            reply = {"intent": "search"}
        elif "query" in fields:
            data = json.loads(payload["messages"][1]["content"])
            reply = {"query": data["query"]}
        elif "selected_positions" in fields:
            reply = {"selected_positions": [1], "reason_codes": []}
        else:
            reply = {"sufficient": True, "reason": "relevant", "rewrite": None}
        return httpx.Response(200, json={"message": {"content": json.dumps(reply)}})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        service = SearchService(FakeEngine(), LocalLLM(client=client))
        first, second = SessionState(), SessionState()
        calls = [
            pool.submit(service.handle, "phone", first),
            pool.submit(service.handle, "handset", second),
        ]
        for task in calls:
            result = task.result(timeout=8)
            assert result.status == "ok" and result.diagnostics.ollama_calls == 4
        assert first.semantic_query == "phone" and second.semantic_query == "handset"
