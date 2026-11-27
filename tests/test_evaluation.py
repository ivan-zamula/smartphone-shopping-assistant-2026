"""Synthetic, independent checks of evaluation math and labels."""

import math
from types import SimpleNamespace

from smartphone_shopping_assistant.evaluation import (
    Judgment,
    build_cases,
    evaluate,
    percentile,
    score_ranking,
)


def test_known_grades_and_unjudged():
    result = score_ranking(
        {"a": 2, "b": 1, "z": 0}, ["a", "unknown", "z", "b"], positive=True, k=3, n=4
    )
    assert result["recall"] == 1
    assert result["precision_judged"] == 0.5
    assert result["judged_at_k"] == 2
    assert result["unjudged_at_k"] == 1
    assert 0 < result["ndcg"] < 1


def test_empty_and_percentiles():
    result = score_ranking({"a": 2}, [], positive=True)
    assert result["false_empty"] and result["recall"] == 0
    assert result["precision_judged"] is None
    assert score_ranking({}, [], positive=False)["ndcg"] is None
    assert percentile([1, 2, 3], 95) == 2.9


def test_duplicate_positive_contributes_once_to_ndcg():
    result = score_ranking({"a": 2, "b": 1}, ["a", "a", "b"], positive=True)
    assert result["ndcg"] == 1
    assert result["recall"] == 1
    assert result["judged_at_k"] == 2
    assert result["ndcg"] <= 1


def test_unjudged_nonempty_is_not_impossible_error():
    unjudged = score_ranking({}, ["u"], positive=False)
    assert unjudged["recall"] is None
    assert unjudged["ndcg"] is None
    assert not unjudged["nonempty_impossible"]
    assert score_ranking({}, ["u"], positive=False, expected_empty=True)["nonempty_impossible"]


def test_retrieval_recall_survives_selection_and_duplicates():
    def candidate(item_id):
        return SimpleNamespace(product=SimpleNamespace(id=item_id))

    class Engine:
        def retrieve(self, request, candidate_limit):
            assert candidate_limit == 20
            return [candidate("a"), candidate("b")]

        def _matches(self, product, filters):
            return True

    case = Judgment("E01", "exact_model", "phone", {}, {"a": 2, "b": 1}, "", "")
    rows, summary = evaluate(
        [case],
        {"a", "b"},
        Engine(),
        ranker=lambda request, candidates: [candidates[0], candidates[0]],
    )
    assert rows[0]["retrieval_recall"] == 1
    assert rows[0]["recall"] == 0.5
    assert rows[0]["violations"] == 1
    assert rows[0]["ndcg"] == 3 / (3 + 1 / math.log2(3))
    assert summary["retrieval_recall"] == 1
    assert summary["selected_recall"] == 0.5


def test_optional_stage_timings_keep_default_engine_api():
    class Engine:
        def retrieve(self, request, candidate_limit, *, timings):
            assert candidate_limit == 20
            timings.update(sql_ms=1.0, embedding_ms=2.0, qdrant_ms=3.0, hydration_ms=4.0)
            return []

        def _matches(self, product, filters):
            return True

    case = Judgment("I01", "impossible", "none", {}, {}, "", "")
    rows, summary = evaluate([case], set(), Engine(), collect_timings=True)
    assert rows[0]["sql_ms"] == 1.0
    assert summary["embedding_ms_cold"] == 2.0
    assert summary["qdrant_ms_cold"] == 3.0
    assert summary["hydration_ms_cold"] == 4.0
    assert summary["llm_ms_cold"] is None


def test_cases_are_source_derived():
    cards = [
        {
            "id": str(i),
            "name": f"Phone variant {i}",
            "brand": "Acme",
            "model": f"Variant {i}",
            "ram_gb": 8,
            "storage_gb": 128,
            "price": 100 + i,
        }
        for i in range(16)
    ]
    cases = build_cases(cards)
    assert len(cases) == 30
    assert {c.category for c in cases} == {
        "exact_model",
        "combination",
        "paraphrase",
        "unsupported",
        "impossible",
    }
    assert all(c.grades for c in cases[:24])
    assert all(not c.grades for c in cases[24:])
    assert all(
        c.query.startswith(("Ищу", "Покажи", "Есть", "Подбери", "Нужен", "Какие", "Найди"))
        for c in cases[16:24]
    )
    assert len(cases[17].grades) == 16  # Every matching source row, not just the seed.
