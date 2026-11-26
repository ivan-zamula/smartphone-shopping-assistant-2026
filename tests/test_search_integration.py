"""Opt-in search against disposable PostgreSQL and in-memory Qdrant."""

import os
from uuid import uuid4

import numpy as np
import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row
from qdrant_client import QdrantClient

from smartphone_shopping_assistant import ingest, search
from smartphone_shopping_assistant.contracts import SearchRequest
from smartphone_shopping_assistant.prepare import make_card

pytestmark = pytest.mark.skipif(
    os.getenv("SSA_RUN_INTEGRATION") != "1", reason="opt-in synthetic integration"
)


class Encoder:
    def encode(self, query, **kwargs):
        assert query.startswith("query: ")
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1
        return vector


@pytest.fixture
def catalog(monkeypatch):
    base = ingest.Settings.from_env()
    dbname = "ssa_search_test_" + uuid4().hex
    with psycopg.connect(**base.postgres, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
        qdrant = QdrantClient(":memory:")
        actual_close = qdrant.close
        monkeypatch.setattr(qdrant, "close", lambda: None)
        monkeypatch.setattr(search, "QdrantClient", lambda **kwargs: qdrant)
        settings = ingest.Settings(base.postgres | {"dbname": dbname}, ":memory:")
        try:
            cards = [
                make_card(
                    {
                        "Smartphone": f"Synthetic {i}",
                        "Brand": brand,
                        "Model": "Phone",
                        "RAM": ram,
                        "Storage": "256",
                        "Color": "Black",
                        "Free": "Yes",
                        "Final Price": price,
                    }
                )
                for i, (brand, ram, price) in enumerate(
                    [("Apple", "8", "100"), ("Apple", "", "200"), ("Samsung", "12", "300")]
                )
            ]
            with psycopg.connect(**settings.postgres, row_factory=dict_row) as conn:
                ingest.write_postgres(conn, cards)
            vectors = np.zeros((3, 384), dtype=np.float32)
            vectors[0, 0] = vectors[1, 1] = vectors[2, 2] = 1
            ingest.write_qdrant(qdrant, cards, vectors, False)
            yield settings, cards
        finally:
            actual_close()
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(dbname)))


def test_search_filters_sql_and_hydrates(catalog):
    settings, cards = catalog
    engine = search.SearchEngine(settings=settings, encoder=Encoder())
    request = SearchRequest.model_validate(
        {
            "query": "Apple 256",
            "filters": {
                "brands": ["Apple"],
                "price": {"maximum": "200", "include_maximum": False},
                "ram_gb": {"minimum": 8},
            },
        }
    )
    result = engine.retrieve(request)
    assert [str(item.product.id) for item in result] == [cards[0]["id"]]
    assert result[0].product.price == 100
    assert engine.get_products([result[0].product.id])[result[0].product.id].brand == "Apple"
    assert (
        engine.retrieve(
            SearchRequest.model_validate(
                {"query": "Apple", "filters": {"brands": ["Apple"], "ram_gb": {"minimum": 9}}}
            )
        )
        == []
    )
