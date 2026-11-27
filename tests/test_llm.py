import json
import time
from decimal import Decimal
from uuid import UUID

import httpx
import pytest

from smartphone_shopping_assistant.contracts import ProductCard, SearchCandidate
from smartphone_shopping_assistant.llm import LLMUnavailable, LocalLLM, literal_brands


def test_literal_brands_boundaries_transliteration_and_or():
    brands = ["Samsung", "OPPO", "Realme"]
    assert literal_brands("Самсунг с 8 ГБ", brands) == (["Samsung"], False)
    assert literal_brands("Samsung или OPPO", brands) == (["Samsung", "OPPO"], False)
    assert literal_brands("не Samsung", brands)[1]
    assert literal_brands("Samsung кроме OPPO", brands)[1]
    assert literal_brands("Samsungish", brands) == ([], False)


def test_memory_number_cannot_authorize_hallucinated_price():
    answer = json.dumps(
        {"intent": "search", "filter_changes": {"price": {"maximum": 8}, "ram_gb": {"maximum": 8}}}
    )
    parsed = LocalLLM(client=client([answer])).parse("Samsung с 8 ГБ ОЗУ")
    assert "price" not in parsed.filter_changes.model_fields_set
    assert parsed.filter_changes.ram_gb.minimum == 8


ID = UUID("00000000-0000-0000-0000-000000000001")


def candidate():
    return SearchCandidate(
        product=ProductCard(
            id=ID,
            name="Synthetic Phone",
            brand="Test",
            model="One",
            ram_gb=8,
            storage_gb=128,
            color="black",
            carrier_unlocked=True,
            price=Decimal("100"),
            currency="USD",
            currency_status="unconfirmed",
            source_currency=None,
            conversion_applied=False,
            raw_fields={},
            text="synthetic",
        ),
        vector_score=0.5,
    )


def client(contents):
    iterator = iter(contents)

    def handler(request):
        payload = json.loads(request.content)
        assert payload["think"] is False
        assert payload["options"]["temperature"] == 0
        return httpx.Response(200, json={"message": {"content": next(iterator)}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parse_patch_omission_and_explicit_clear():
    llm = LocalLLM(
        client=client(
            [
                json.dumps(
                    {
                        "intent": "refine",
                        "filter_changes": {"price": None},
                        "clear_fields": ["price"],
                    }
                ),
                json.dumps({"intent": "search", "filter_changes": {"brands": ["Test"]}}),
            ]
        )
    )
    clear = llm.parse("clear price")
    assert clear.filter_changes.model_fields_set == {"price"}
    assert clear.filter_changes.price is None
    update = llm.parse("Test")
    assert update.filter_changes.model_fields_set == {"brands"}


def test_invalid_json_one_retry():
    llm = LocalLLM(client=client(["broken", json.dumps({"intent": "search"})]))
    assert llm.parse("phone").intent == "search"
    with pytest.raises(LLMUnavailable, match="Invalid"):
        LocalLLM(client=client(["broken", "broken"])).parse("phone")


@pytest.mark.parametrize(
    "ids",
    [
        [1, 1],
        [2],
    ],
)
def test_reject_duplicate_or_foreign_id(ids):
    answer = json.dumps({"selected_positions": ids, "reason_codes": []})
    with pytest.raises(LLMUnavailable, match="selection"):
        LocalLLM(client=client([answer, answer])).select("phone", [candidate()])


def test_captured_selection_normalizes_reason_codes_without_retry():
    base = candidate()
    pool = [
        base.model_copy(
            update={
                "product": base.product.model_copy(
                    update={"id": UUID(int=i), "name": f"Synthetic Phone {i}"}
                )
            }
        )
        for i in range(1, 11)
    ]
    answer = json.dumps(
        {
            "selected_positions": [1, 4, 5, 6, 7],
            "reason_codes": ["price", "memory", "price", "price", "price"],
        }
    )
    # Only one mock response exists: an unnecessary retry would fail this test.
    choice = LocalLLM(client=client([answer])).select("synthetic phone", pool)
    assert choice.selected_ids == [UUID(int=i) for i in (1, 4, 5, 6, 7)]
    assert choice.reason_codes == ["price", "memory"]
    assert choice.abstain is False


def test_reason_code_normalization_still_rejects_unknown_codes():
    answer = json.dumps({"selected_positions": [1], "reason_codes": ["price", "unknown", "price"]})
    with pytest.raises(LLMUnavailable, match="selection"):
        LocalLLM(client=client([answer, answer])).select("phone", [candidate()])


def test_selection_schema_and_validation_feedback_retry():
    requests = []
    responses = iter(
        [
            {"selected_positions": [1, 1], "reason_codes": []},
            {"selected_positions": [1], "reason_codes": []},
        ]
    )

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"message": {"content": json.dumps(next(responses))}})

    choice = LocalLLM(client=httpx.Client(transport=httpx.MockTransport(handler))).select(
        "phone", [candidate()], limit=1
    )
    assert choice.selected_ids == [ID]
    schema = requests[0]["format"]
    assert set(schema["required"]) == {"selected_positions", "reason_codes"}
    assert schema["properties"]["selected_positions"]["items"]["enum"] == [1]
    assert schema["properties"]["selected_positions"]["maxItems"] == 1
    assert requests[0]["options"]["num_ctx"] == 4096
    assert "duplicate" in json.loads(requests[1]["messages"][1]["content"])["validation_feedback"]


def test_deadline_exceeded_before_request():
    with pytest.raises(LLMUnavailable, match="deadline"):
        LocalLLM(client=client([])).parse("phone", deadline=time.monotonic() - 1)


@pytest.mark.parametrize(
    ("query", "field", "expected"),
    [
        ("RAM 8 GB", "ram_gb", {"minimum": 8, "maximum": 8}),
        ("128GB", "storage_gb", {"minimum": 128, "maximum": 128}),
        ("Samsung RAM >= 8 GB storage >= 128 GB", "storage_gb", {"minimum": 128}),
        ("не менее 8 ГБ ОЗУ", "ram_gb", {"minimum": 8}),
        ("under 128 GB storage", "storage_gb", {"maximum": 128, "include_maximum": False}),
        ("менее 128 ГБ памяти", "storage_gb", {"maximum": 128, "include_maximum": False}),
        ("at most 128 GB storage", "storage_gb", {"maximum": 128}),
        ("не более 128 ГБ памяти", "storage_gb", {"maximum": 128}),
    ],
)
def test_memory_evidence_corrects_llm_direction(query, field, expected):
    wrong = json.dumps(
        {
            "intent": "search",
            "filter_changes": {field: {"maximum": 128}},
        }
    )
    parsed = LocalLLM(client=client([wrong])).parse(query)
    actual = getattr(parsed.filter_changes, field)
    for key, value in expected.items():
        assert getattr(actual, key) == value
    if "minimum" not in expected:
        assert actual.minimum is None
    if "maximum" not in expected:
        assert actual.maximum is None


@pytest.mark.parametrize(
    "query",
    [
        "Подбери Motorola: ОЗУ 8 ГБ, накопитель 256 ГБ",
        "Find Motorola: RAM 8 GB, storage 256 GB",
        "Find Motorola: 8 GB RAM, 256 GB storage",
        "Подбери Motorola: 8 ГБ ОЗУ; 256 ГБ накопитель",
    ],
)
def test_memory_evidence_respects_comma_clause_boundary(query):
    from smartphone_shopping_assistant.llm import _memory_evidence

    assert _memory_evidence(query) == {
        "ram_gb": {"minimum": 8, "maximum": 8},
        "storage_gb": {"minimum": 256, "maximum": 256},
    }


def test_memory_evidence_keeps_conflicting_amounts_ambiguous():
    from smartphone_shopping_assistant.llm import _memory_evidence

    with pytest.raises(ValueError, match="ambiguous memory amounts"):
        _memory_evidence("RAM 8 GB and 12 GB RAM")


def test_russian_brand_and_both_memory_bounds():
    answer = json.dumps(
        {
            "intent": "search",
            "filter_changes": {
                "brands": ["Samsung"],
                "ram_gb": {"maximum": 8},
                "storage_gb": {"maximum": 128},
            },
        }
    )
    parsed = LocalLLM(client=client([answer])).parse(
        "Самсунг с ОЗУ не менее 8 ГБ и памятью не менее 128 ГБ"
    )
    assert parsed.filter_changes.brands == ["Samsung"]
    assert parsed.filter_changes.ram_gb.minimum == 8
    assert parsed.filter_changes.storage_gb.minimum == 128
    assert parsed.filter_changes.ram_gb.maximum is None
    assert parsed.filter_changes.storage_gb.maximum is None


def test_unmentioned_memory_filter_is_removed():
    answer = json.dumps(
        {
            "intent": "search",
            "filter_changes": {
                "storage_gb": {"maximum": 128},
            },
        }
    )
    parsed = LocalLLM(client=client([answer])).parse("Samsung phone")
    assert "storage_gb" not in parsed.filter_changes.model_fields_set


def test_compare_first_two_positions():
    answer = json.dumps({"intent": "compare", "previous_result_positions": [1]})
    parsed = LocalLLM(client=client([answer])).parse("Сравни первые два")
    assert parsed.previous_result_positions == [1, 2]
