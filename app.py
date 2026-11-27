"""Local Streamlit interface for the read-only shopping dialogue."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import streamlit as st

from smartphone_shopping_assistant.contracts import FilterPatch, ProductCard
from smartphone_shopping_assistant.dialogue import SearchService, SessionState
from smartphone_shopping_assistant.i18n import translate
from smartphone_shopping_assistant.llm import MODEL


def _t(text: str) -> str:
    language = st.session_state.get("ui_language", st.query_params.get("lang", "ru"))
    return translate(text, language)


def _range(minimum: str, maximum: str, *, integer: bool) -> dict | None:
    if not minimum.strip() and not maximum.strip():
        return None
    convert = int if integer else Decimal
    try:
        values = [convert(value.strip()) if value.strip() else None for value in (minimum, maximum)]
        if integer and any(
            value is not None and str(value) != source.strip()
            for value, source in zip(values, (minimum, maximum), strict=True)
        ):
            raise ValueError
    except (ValueError, InvalidOperation) as exc:
        raise ValueError("Укажите целые ГБ и числовую цену.") from exc
    return {"minimum": values[0], "maximum": values[1]}


def _ui_values() -> dict[str, str]:
    return {
        key: st.session_state.get("ui_" + key, "Любой" if key == "unlocked" else "")
        for key in (
            "brands",
            "price_min",
            "price_max",
            "ram_min",
            "ram_max",
            "storage_min",
            "storage_max",
            "unlocked",
        )
    }


def _patch(now: dict[str, str], before: dict[str, str]) -> FilterPatch:
    changes = {}
    if now["brands"] != before["brands"]:
        changes["brands"] = [s.strip() for s in now["brands"].split(",") if s.strip()] or None
    for field, a, b, integer in (
        ("price", "price_min", "price_max", False),
        ("ram_gb", "ram_min", "ram_max", True),
        ("storage_gb", "storage_min", "storage_max", True),
    ):
        if now[a] != before[a] or now[b] != before[b]:
            changes[field] = _range(now[a], now[b], integer=integer)
    if now["unlocked"] != before["unlocked"]:
        changes["carrier_unlocked"] = {"Любой": None, "Да": True, "Нет": False}[now["unlocked"]]
    return FilterPatch.model_validate(changes)


def _memory(value: int | None) -> str:
    return "—" if value is None else f"{value} {_t('ГБ')}"


def _price(card: ProductCard) -> str:
    return f"{card.price} {card.currency}*"


def _product(card: ProductCard, number: int) -> None:
    st.markdown(f"**{number}. {card.name}** · {_price(card)}")
    st.caption(
        f"{card.brand} · {_t('ОЗУ')} {_memory(card.ram_gb)} "
        f"· {_t('Память')} {_memory(card.storage_gb)} "
        f"· {_t('Разблокирован' if card.carrier_unlocked else 'Привязан к оператору')}"
    )


def _show(response) -> None:
    if response.status == "error":
        st.error(_t(response.message or "Поиск недоступен."))
    elif response.status == "needs_clarification":
        st.info(_t(response.message or "Уточните запрос."))
    elif response.status == "empty":
        st.warning(_t(response.message or "Ничего не найдено."))
    elif response.message:
        st.write(_t(response.message))
    for number, card in enumerate(response.items[:5], 1):
        _product(card, number)
    if response.comparison:
        st.table(
            {
                _t("Характеристика"): [
                    _t("Модель"),
                    _t("Бренд"),
                    _t("ОЗУ"),
                    _t("Память"),
                    _t("Цвет"),
                    _t("Без блокировки"),
                    _t("Цена*"),
                ],
                "1 · " + response.comparison[0].name: [
                    response.comparison[0].model,
                    response.comparison[0].brand,
                    _memory(response.comparison[0].ram_gb),
                    _memory(response.comparison[0].storage_gb),
                    response.comparison[0].color,
                    _t("Да" if response.comparison[0].carrier_unlocked else "Нет"),
                    _price(response.comparison[0]),
                ],
                "2 · " + response.comparison[1].name: [
                    response.comparison[1].model,
                    response.comparison[1].brand,
                    _memory(response.comparison[1].ram_gb),
                    _memory(response.comparison[1].storage_gb),
                    response.comparison[1].color,
                    _t("Да" if response.comparison[1].carrier_unlocked else "Нет"),
                    _price(response.comparison[1]),
                ],
            }
        )


@st.cache_resource
def _service() -> SearchService:
    return SearchService()


def _clear_ui() -> None:
    st.session_state.clear_previous_values = _ui_values()
    for key in st.session_state.last_ui_values:
        st.session_state["ui_" + key] = "Любой" if key == "unlocked" else ""
    st.session_state.pending_clear = True


def _applied_filters(filters) -> list[str]:
    lines = []
    if filters.brands:
        lines.append(_t("Бренды") + ": " + ", ".join(filters.brands))
    for label, bounds, unit in (
        ("Цена", filters.price, " USD"),
        ("ОЗУ", filters.ram_gb, " " + _t("ГБ")),
        ("Память", filters.storage_gb, " " + _t("ГБ")),
    ):
        label = _t(label)
        if bounds:
            if bounds.minimum is not None and bounds.maximum is not None:
                lines.append(f"{label}: {bounds.minimum}–{bounds.maximum}{unit}")
            elif bounds.minimum is not None:
                lines.append(f"{label}: {_t('от')} {bounds.minimum}{unit}")
            elif bounds.maximum is not None:
                lines.append(f"{label}: {_t('до')} {bounds.maximum}{unit}")
    if filters.carrier_unlocked is not None:
        lines.append(
            _t("Без блокировки оператора") + ": " + _t("Да" if filters.carrier_unlocked else "Нет")
        )
    return lines or [_t("Фильтры не заданы")]


def main() -> None:
    st.set_page_config(page_title=_t("Подбор смартфона"), page_icon="📱", layout="wide")
    language = st.sidebar.selectbox(
        "Language",
        ["ru", "en"],
        index=1 if st.query_params.get("lang") == "en" else 0,
        format_func={"ru": "Русский", "en": "English"}.get,
        key="ui_language",
    )
    if st.query_params.get("lang") != st.session_state.ui_language:
        st.query_params["lang"] = st.session_state.ui_language
    st.title(_t("📱 Подбор смартфона"))
    st.caption(
        _t("Локальная модель: {model}. Поиск по каталогу, до 5 товаров в ответе.").format(
            model=MODEL
        )
    )
    st.info(_t("* USD — валюта проекта. Валюта исходного каталога не подтверждена; пересчёта нет."))
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = SessionState()
        st.session_state.history = []
        st.session_state.last_ui_values = _ui_values()
    if "restore_ui_values" in st.session_state:
        for key, value in st.session_state.pop("restore_ui_values").items():
            st.session_state["ui_" + key] = value

    with st.sidebar:
        st.header(_t("Фильтры"))
        st.button(_t("Очистить фильтры"), on_click=_clear_ui)
        with st.form("filters"):
            st.text_input(_t("Бренды через запятую"), key="ui_brands")
            st.text_input(_t("Цена от"), key="ui_price_min")
            st.text_input(_t("Цена до"), key="ui_price_max")
            st.text_input(_t("ОЗУ от, ГБ"), key="ui_ram_min")
            st.text_input(_t("ОЗУ до, ГБ"), key="ui_ram_max")
            st.text_input(_t("Память от, ГБ"), key="ui_storage_min")
            st.text_input(_t("Память до, ГБ"), key="ui_storage_max")
            st.selectbox(
                _t("Без блокировки оператора"),
                ["Любой", "Да", "Нет"],
                key="ui_unlocked",
                format_func=lambda value: translate(value, language),
            )
            apply = st.form_submit_button(_t("Применить фильтры"))
        st.subheader(_t("Применено"))
        filters = st.session_state.dialogue.filters
        for line in _applied_filters(filters):
            st.write(line)

    action = None
    patch = None
    if st.session_state.pop("pending_clear", False):
        patch = FilterPatch(
            brands=None, price=None, ram_gb=None, storage_gb=None, carrier_unlocked=None
        )
        action = ""
    elif apply:
        try:
            now = _ui_values()
            patch = _patch(now, st.session_state.last_ui_values)
            action = ""
        except ValueError as exc:
            st.sidebar.error(_t(str(exc)))
    left, right = st.columns(2)
    if left.button(_t("Показать дешевле"), use_container_width=True):
        action, patch = _t("Покажи дешевле первого"), None
    if right.button(_t("Сравнить первые два"), use_container_width=True):
        action, patch = _t("Сравни первые два телефона"), None
    query = st.chat_input(_t("Например: Samsung с 8 ГБ ОЗУ"))
    if query:
        action, patch = query, None
    history = st.session_state.history
    if len(history) > 1:
        with st.expander(_t("История диалога — предыдущие ответы"), expanded=False):
            for question, answer in history[:-1]:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    _show(answer)
    for question, answer in history[-1:]:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            _show(answer)
    if action is not None:
        with st.spinner(_t("Подбираем смартфоны…")):
            response = _service().handle(action, st.session_state.dialogue, ui_changes=patch)
        if response.status in ("ok", "empty"):
            st.session_state.last_ui_values = _ui_values()
        elif st.session_state.get("pending_clear") or "clear_previous_values" in st.session_state:
            st.session_state.restore_ui_values = st.session_state.clear_previous_values
        st.session_state.pop("clear_previous_values", None)
        st.session_state.history.append((action or _t("Фильтры"), response))
        st.rerun()


if __name__ == "__main__":
    main()
