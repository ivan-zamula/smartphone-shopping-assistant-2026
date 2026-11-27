"""Synthetic, isolated dialogue state and comparison checks."""

from decimal import Decimal
from uuid import UUID

from smartphone_shopping_assistant.contracts import (
    FilterPatch,
    NumericRange,
    ParsedIntent,
    ProductCard,
    SearchCandidate,
)
from smartphone_shopping_assistant.dialogue import SearchService, SessionState
from smartphone_shopping_assistant.llm import Selection
from smartphone_shopping_assistant.search import SearchEngine


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

    def retrieve(self, request, candidate_limit):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("database down")
        return [
            SearchCandidate(product=c, vector_score=0.8)
            for c in self.products.values()
            if self._matches(c, request.filters)
        ]

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

    def parse(self, query, contextual_info, deadline):
        return self.next_intent

    def select(self, query, candidates, limit, deadline):
        self.select_calls += 1
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


def test_compare_timeout_after_hydration(monkeypatch):
    service, engine, llm, session = setup()
    session.last_successful_ids = [UUID(int=1), UUID(int=2)]
    llm.next_intent = ParsedIntent(intent="compare", previous_result_positions=[1, 2])
    ticks = iter([0, 0, 0, 181])
    monkeypatch.setattr("smartphone_shopping_assistant.workflow.monotonic", lambda: next(ticks))
    result = service.handle("compare", session)
    assert result.status == "error" and not result.items and not result.comparison
    assert session.last_successful_ids == [UUID(int=1), UUID(int=2)]


def test_search_timeout_after_final_hydration(monkeypatch):
    service, engine, llm, session = setup()
    ticks = iter([0, 0, 0, 0, 0, 0, 181])
    monkeypatch.setattr("smartphone_shopping_assistant.workflow.monotonic", lambda: next(ticks))
    result = service.handle("phone", session)
    assert result.status == "error" and not result.items and not result.comparison
    assert session.last_successful_ids == [] and session.semantic_query == ""
