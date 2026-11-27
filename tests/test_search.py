"""Focused retrieval contract checks with synthetic IDs and vectors."""

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from qdrant_client import models

from smartphone_shopping_assistant import search
from smartphone_shopping_assistant.contracts import SearchFilters, SearchRequest
from smartphone_shopping_assistant.ingest import DIMENSION, MODEL, REVISION

A, B, FOREIGN = uuid4(), uuid4(), uuid4()


class Encoder:
    def __init__(self, vector=None):
        self.vector = vector if vector is not None else np.eye(1, DIMENSION)[0]

    def encode(self, *args, **kwargs):
        assert args[0].startswith("query: ")
        return self.vector


class FakeQdrant:
    def __init__(self, hits, config=None):
        self.hits = hits
        self.config = config or models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE)
        self.query = None
        self.closed = False

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self.config)))

    def query_points(self, **kwargs):
        self.query = kwargs
        return SimpleNamespace(points=self.hits)

    def close(self):
        self.closed = True


def hit(id, score=0.8, payload=None):
    return SimpleNamespace(
        id=id,
        score=score,
        payload=payload
        if payload is not None
        else {
            "embedding_model": MODEL,
            "embedding_revision": REVISION,
        },
    )


def engine(monkeypatch, ids, hits, products=None, config=None):
    q = FakeQdrant(hits, config)
    monkeypatch.setattr(search, "QdrantClient", lambda **kwargs: q)
    e = search.SearchEngine(
        settings=SimpleNamespace(postgres={}, qdrant_url=":memory:"), encoder=Encoder()
    )
    monkeypatch.setattr(e, "_eligible", lambda filters: ids)
    monkeypatch.setattr(
        e,
        "get_products",
        lambda selected: products if products is not None else {i: {"id": i} for i in selected},
    )
    return e, q


def test_nonempty_mixed_ids_and_stable_tie(monkeypatch):
    # Candidate validation is exercised with real ProductCard construction elsewhere.
    e, q = engine(monkeypatch, [A, B], [hit(str(B)), hit(A)])
    monkeypatch.setattr(search, "SearchCandidate", lambda **kw: kw)
    monkeypatch.setattr(e, "_matches", lambda product, filters: True)
    result = e.retrieve(SearchRequest(query="phone"))
    assert [x["product"]["id"] for x in result] == sorted([A, B], key=str)
    assert set(q.query["query_filter"].must[0].has_id) == {str(A), str(B)}
    assert q.closed


def test_no_candidates_skip_encoder(monkeypatch):
    e, _ = engine(monkeypatch, [], [])
    monkeypatch.setattr(e, "get_products", lambda _: pytest.fail("hydration called"))
    assert e.retrieve(SearchRequest(query="phone")) == []


@pytest.mark.parametrize("vector", [np.zeros(DIMENSION), np.full(DIMENSION, np.nan), np.ones(3)])
def test_bad_vector(monkeypatch, vector):
    e, _ = engine(monkeypatch, [A], [hit(A)])
    e.encoder = Encoder(vector)
    with pytest.raises(search.SearchUnavailable, match="Query encoder"):
        e.retrieve(SearchRequest(query="phone"))


@pytest.mark.parametrize("hits", [[hit(FOREIGN)], [hit(A), hit(str(A))], [hit(A, payload={})]])
def test_reject_bad_points(monkeypatch, hits):
    e, _ = engine(monkeypatch, [A], hits)
    with pytest.raises(search.SearchUnavailable, match="Qdrant"):
        e.retrieve(SearchRequest(query="phone"))


def test_wrong_collection_and_missing_product(monkeypatch):
    e, _ = engine(
        monkeypatch,
        [A],
        [hit(A)],
        config=models.VectorParams(size=384, distance=models.Distance.DOT),
    )
    with pytest.raises(search.SearchUnavailable, match="Qdrant"):
        e.retrieve(SearchRequest(query="phone"))
    e, _ = engine(monkeypatch, [A], [hit(A)], products={})
    with pytest.raises(search.SearchUnavailable, match="hydration"):
        e.retrieve(SearchRequest(query="phone"))


def test_sql_filter_parameterization_and_boundaries(monkeypatch):
    calls = []

    class Conn:
        def execute(self, query, params):
            calls.append((query, params))
            return SimpleNamespace(fetchall=lambda: [(A,)])

        def close(self):
            pass

    e = search.SearchEngine(settings=SimpleNamespace(postgres={}, qdrant_url=""), encoder=Encoder())
    monkeypatch.setattr(e, "_connect", Conn)
    malicious = "Apple'); DROP TABLE products; --"
    filters = SearchFilters.model_validate(
        {
            "brands": [malicious],
            "price": {"minimum": "100", "include_minimum": False},
            "ram_gb": {"maximum": 8, "include_maximum": False},
            "carrier_unlocked": False,
        }
    )
    assert e._eligible(filters) == [A]
    query, params = calls[0]
    assert malicious not in query and params[0] == [malicious.lower()]
    assert "price > %s" in query and "ram_gb < %s" in query
    assert "LIMIT" not in query and "carrier_unlocked = %s" in query
    assert params[1:] == [Decimal("100"), 8, False]


def test_known_brands_reads_current_catalogue(monkeypatch):
    calls = []

    class Conn:
        def execute(self, query):
            calls.append(query)
            return SimpleNamespace(fetchall=lambda: [("OPPO",), ("Samsung",)])

        def close(self):
            pass

    e = search.SearchEngine(settings=SimpleNamespace(postgres={}, qdrant_url=""))
    monkeypatch.setattr(e, "_connect", Conn)
    assert e.known_brands() == ["OPPO", "Samsung"]
    assert calls == ["SELECT DISTINCT brand FROM public.products ORDER BY brand"]
