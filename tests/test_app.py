"""Synthetic UI checks; no catalog or local model is required."""

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from streamlit.testing.v1 import AppTest

from smartphone_shopping_assistant.contracts import ProductCard, SearchFilters
from smartphone_shopping_assistant.dialogue import DialogueResponse


def _response(status="error", message="Поиск сейчас недоступен."):
    return DialogueResponse(
        request_id=uuid4(), status=status, message=message, filters=SearchFilters()
    )


def test_app_has_local_model_currency_and_independent_sessions():
    first = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    second = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not first.exception and not second.exception
    assert first.session_state.dialogue is not second.session_state.dialogue
    assert "qwen3:1.7b" in first.caption[0].value
    assert "не подтверждена" in first.info[0].value


def test_filter_application_only_sends_changed_fields_and_error_is_fresh():
    calls = []

    def handle(self, text, session, ui_changes=None):
        calls.append((text, ui_changes))
        return _response()

    with patch("smartphone_shopping_assistant.dialogue.SearchService.handle", handle):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        app.text_input(key="ui_brands").set_value("Samsung")
        app.button(key="FormSubmitter:filters-Применить фильтры").click().run()
        assert calls[-1][0] == ""
        assert calls[-1][1].model_fields_set == {"brands"}
        assert calls[-1][1].brands == ["Samsung"]
        assert "недоступен" in app.error[0].value
        app.button(key="FormSubmitter:filters-Применить фильтры").click().run()
        assert calls[-1][1].brands == ["Samsung"]
        assert len(app.session_state.history) == 2


def test_clear_failure_restores_fields_and_retries():
    calls = []

    def handle(self, text, session, ui_changes=None):
        calls.append(ui_changes)
        return _response("ok" if len(calls) == 1 else "error")

    with patch("smartphone_shopping_assistant.dialogue.SearchService.handle", handle):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        app.text_input(key="ui_brands").set_value("Samsung")
        app.button(key="FormSubmitter:filters-Применить фильтры").click().run()
        app.button[2].click().run()
        assert calls[-1].brands is None
        assert app.text_input(key="ui_brands").value == "Samsung"
        assert app.session_state.last_ui_values["brands"] == "Samsung"


def test_readable_filters_and_numbered_cards():
    from decimal import Decimal

    from smartphone_shopping_assistant.contracts import ProductCard

    card = ProductCard.model_validate(
        {
            "id": str(uuid4()),
            "name": "Sample",
            "brand": "Acme",
            "model": "S",
            "price": Decimal("100"),
            "currency": "USD",
            "currency_status": "project_assumed",
            "source_currency": None,
            "conversion_applied": False,
            "raw_fields": {},
            "text": "Sample",
            "carrier_unlocked": True,
            "ram_gb": 8,
            "storage_gb": 128,
            "color": "black",
        }
    )

    def handle(self, text, session, ui_changes=None):
        session.filters = SearchFilters(brands=["Acme"])
        return DialogueResponse(
            request_id=uuid4(), status="ok", filters=session.filters, items=[card]
        )

    with patch("smartphone_shopping_assistant.dialogue.SearchService.handle", handle):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        app.text_input(key="ui_brands").set_value("Acme")
        app.button(key="FormSubmitter:filters-Применить фильтры").click().run()
        assert any("Бренды: Acme" in x.value for x in app.sidebar.markdown)
        assert any("1. Sample" in x.value for x in app.markdown)


def test_previous_cards_are_collapsed_after_new_error():
    from decimal import Decimal

    from smartphone_shopping_assistant.contracts import ProductCard

    card = ProductCard(
        id=uuid4(),
        name="Old sample",
        brand="Acme",
        model="S",
        ram_gb=8,
        storage_gb=128,
        color="black",
        carrier_unlocked=True,
        price=Decimal("100"),
        currency="USD",
        currency_status="project_assumed",
        source_currency=None,
        conversion_applied=False,
        raw_fields={},
        text="synthetic",
    )
    calls = []

    def handle(self, text, session, ui_changes=None):
        calls.append(text)
        if len(calls) == 1:
            return DialogueResponse(request_id=uuid4(), status="ok", items=[card])
        return _response()

    with patch("smartphone_shopping_assistant.dialogue.SearchService.handle", handle):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        app.button[1].click().run()
        app.button[1].click().run()
        assert len(app.session_state.history) == 2
        assert app.session_state.history[-1][1].status == "error"
        assert len(app.expander) == 1
        assert "История диалога" in app.expander[0].label
        assert app.expander[0].proto.expanded is False


def test_english_locale_translates_errors_and_preserves_filter_values():
    calls = []

    def handle(self, text, session, ui_changes=None):
        calls.append(ui_changes)
        return _response(message="Поиск сейчас недоступен. Повторите запрос позже.")

    with patch("smartphone_shopping_assistant.dialogue.SearchService.handle", handle):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
        app.query_params["lang"] = "en"
        app.run()
        assert not app.exception
        assert app.title[0].value == "📱 Smartphone shopping assistant"
        assert "source currency is unconfirmed" in app.info[0].value
        app.text_input(key="ui_brands").set_value("Samsung")
        app.selectbox(key="ui_unlocked").set_value("Да")
        app.button(key="FormSubmitter:filters-Apply filters").click().run()
        assert calls[-1].brands == ["Samsung"]
        assert calls[-1].carrier_unlocked is True
        assert app.error[0].value == "Search is currently unavailable. Please try again later."

        dialogue = app.session_state.dialogue
        app.selectbox(key="ui_language").set_value("ru").run()
        assert app.session_state.dialogue is dialogue
        assert app.text_input(key="ui_brands").value == "Samsung"
        assert app.selectbox(key="ui_unlocked").value == "Да"
        assert len(calls) == 1  # Changing the language does not repeat a search.
        assert app.title[0].value == "📱 Подбор смартфона"


def test_english_cards_and_comparison_preserve_catalog_facts():
    card = ProductCard.model_validate(
        {
            "id": str(uuid4()),
            "name": "Synthetic Negro Libre",
            "brand": "Acme",
            "model": "S",
            "price": "100.25",
            "currency": "USD",
            "currency_status": "unconfirmed",
            "source_currency": None,
            "conversion_applied": False,
            "raw_fields": {},
            "text": "synthetic",
            "carrier_unlocked": False,
            "ram_gb": None,
            "storage_gb": 128,
            "color": "Negro",
        }
    )
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.query_params["lang"] = "en"
    app.run()
    app.session_state.history = [
        (
            "Show Acme phones",
            DialogueResponse(
                request_id=uuid4(),
                status="ok",
                items=[card],
                message="Товары из каталога; валюта источника не подтверждена, пересчёта нет.",
            ),
        )
    ]
    app.run()
    assert not app.exception
    assert any("Synthetic Negro Libre" in x.value and "100.25 USD" in x.value for x in app.markdown)
    assert any(
        "RAM —" in x.value and "Storage 128 GB" in x.value and "Carrier locked" in x.value
        for x in app.caption
    )
    assert any("Catalog products; source currency unconfirmed" in x.value for x in app.markdown)

    second = card.model_copy(update={"id": uuid4(), "name": "Second synthetic phone"})
    app.session_state.history.append(
        (
            "Compare phones",
            DialogueResponse(
                request_id=uuid4(),
                status="ok",
                comparison=[card, second],
                message="Сравнение по текущим данным каталога.",
            ),
        )
    )
    app.run()
    assert not app.exception
    assert "Feature" in app.table[0].value.columns
    assert app.table[0].value.iloc[4, 1] == "Negro"
    assert app.table[0].value.iloc[6, 1] == "100.25 USD*"


def test_search_diagnostics_collapsed_and_translated():
    from smartphone_shopping_assistant.workflow import WorkflowDiagnostics, WorkflowStep

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    answer = DialogueResponse(
        request_id=uuid4(),
        status="ok",
        steps=[WorkflowStep(node="retrieve", elapsed_ms=82)],
        diagnostics=WorkflowDiagnostics(
            queries=["phone"],
            search_attempts=1,
            ollama_calls=4,
            candidate_count=20,
            stop_reason="sufficient",
        ),
    )
    app.session_state.history = [("phone", answer)]
    app.run()
    assert not app.exception
    panel = app.expander[0]
    assert panel.label == "Как выполнен поиск" and panel.proto.expanded is False
    assert any("Кандидаты: 20" in item.value for item in panel.caption)
    app.selectbox(key="ui_language").set_value("en").run()
    panel = app.expander[0]
    assert panel.label == "Search execution" and panel.proto.expanded is False
    assert any("Ollama calls: 4" in item.value for item in panel.caption)
    assert any("Retrieve candidates" in item.value for item in panel.caption)
    assert panel.text[0].value == "Query 1 · phone"
