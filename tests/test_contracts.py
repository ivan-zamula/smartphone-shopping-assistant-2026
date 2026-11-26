from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from smartphone_shopping_assistant.contracts import (
    FilterPatch,
    IntegerRange,
    NumericRange,
    ParsedIntent,
    ProductCard,
    SearchFilters,
    SearchRequest,
)


def test_decimal_ranges_and_patch_null_semantics():
    bounds = NumericRange(minimum="1.2300", maximum="4")
    assert bounds.minimum == Decimal("1.2300")
    assert FilterPatch().model_fields_set == set()
    assert FilterPatch(price=None).model_fields_set == {"price"}


@pytest.mark.parametrize(
    "factory, data",
    [
        (NumericRange, {"minimum": 2, "maximum": 1}),
        (NumericRange, {"minimum": "NaN"}),
        (NumericRange, {"minimum": 2, "maximum": 2, "include_maximum": False}),
        (NumericRange, {"minimum": "-Infinity"}),
        (IntegerRange, {"minimum": True}),
        (IntegerRange, {"maximum": 2.5}),
        (IntegerRange, {"minimum": 0}),
        (NumericRange, {"minimum": "-1"}),
        (IntegerRange, {"minimum": -1}),
        (IntegerRange, {"minimum": 1.5}),
        (IntegerRange, {"minimum": False}),
    ],
)
def test_invalid_ranges(factory, data):
    with pytest.raises(ValidationError):
        factory(**data)


def test_filters_and_intent_validation():
    with pytest.raises(ValidationError):
        SearchFilters(brands=[" "])
    with pytest.raises(ValidationError):
        SearchFilters(carrier_unlocked="yes")
    for positions in ([1, 1], [0], [True], ["1"], [1.5], list(range(1, 12))):
        with pytest.raises(ValidationError):
            ParsedIntent(intent="compare", previous_result_positions=positions)


def test_request_and_product_validation():
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")
    with pytest.raises(ValidationError):
        SearchRequest(query="phone", limit=11)
    for limit in (True, 2.5):
        with pytest.raises(ValidationError):
            SearchRequest(query="phone", limit=limit)
    assert IntegerRange(minimum=8, maximum=16).minimum == 8
    assert SearchFilters(ram_gb={"minimum": 8}).ram_gb.minimum == 8
    with pytest.raises(ValidationError):
        SearchRequest(query="phone", mode="external")
    with pytest.raises(ValidationError):
        ProductCard(id="bad")
    base = dict(
        id=uuid4(),
        name="x",
        brand="x",
        model="x",
        ram_gb=None,
        storage_gb=64,
        color="black",
        carrier_unlocked=True,
        price="1.20",
        currency="USD",
        currency_status="user_selected",
        source_currency=None,
        conversion_applied=False,
        raw_fields={},
        text="x",
    )
    card = ProductCard(**base)
    assert card.price == Decimal("1.20")
    for changes in ({"ram_gb": True}, {"price": "Infinity"}, {"price": "0"}):
        with pytest.raises(ValidationError):
            ProductCard(**(base | changes))
