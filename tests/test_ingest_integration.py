"""Opt-in PostgreSQL tests in a disposable database with synthetic cards."""

import json
import os
from uuid import uuid4

import numpy as np
import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row
from qdrant_client import QdrantClient, models

from smartphone_shopping_assistant import ingest as module
from smartphone_shopping_assistant.prepare import make_card

pytestmark = pytest.mark.skipif(
    os.environ.get("SSA_RUN_INTEGRATION") != "1",
    reason="Set SSA_RUN_INTEGRATION=1 to use local PostgreSQL and a disposable database",
)


@pytest.fixture
def settings():
    settings = module.Settings.from_env()
    database = "ssa_ingest_test_" + uuid4().hex
    with psycopg.connect(**settings.postgres, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        try:
            yield module.Settings(settings.postgres | {"dbname": database}, ":memory:")
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


def cards():
    return [
        make_card(
            {
                "Smartphone": f"Synthetic Phone {i}",
                "Brand": "Synthetic",
                "Model": "Phone",
                "RAM": "" if i == 0 else "8",
                "Storage": "256",
                "Color": "Blue",
                "Free": "Yes",
                "Final Price": "0.100000000000000001" if i == 0 else "199.99",
            }
        )
        for i in range(2)
    ]


def test_postgres_batch_error_rolls_back_all_changes(settings):
    original = cards()
    with psycopg.connect(**settings.postgres, autocommit=True, row_factory=dict_row) as connection:
        module.write_postgres(connection, original)
        changed = cards()
        changed[0]["brand"] = "Must roll back"
        changed[1]["raw_fields"]["Final Price"] = "-1"
        with pytest.raises(psycopg.errors.CheckViolation):
            module.write_postgres(connection, changed)
        rows = connection.execute("SELECT brand FROM public.products").fetchall()
        assert len(rows) == 2 and all(row["brand"] == "Synthetic" for row in rows)
        module.check_postgres(connection, {card["id"] for card in original})


@pytest.mark.filterwarnings("ignore:Local mode performs exact.*:UserWarning")
def test_partial_qdrant_failure_is_repaired_by_rerun(settings, tmp_path, monkeypatch):
    source = cards()
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(source))
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0] = vectors[1, 1] = 1
    monkeypatch.setattr(module, "encode_cards", lambda _: vectors)
    client = QdrantClient(":memory:")
    original_close = client.close
    monkeypatch.setattr(client, "close", lambda: None)
    monkeypatch.setattr(module, "QdrantClient", lambda **_: client)
    write = module.write_qdrant

    def interrupted(client, cards, vectors, exists):
        write(client, cards[:1], vectors[:1], exists)
        raise ConnectionError("Synthetic interrupted upload")

    monkeypatch.setattr(module, "write_qdrant", interrupted)
    try:
        with pytest.raises(RuntimeError, match="PostgreSQL committed.*NOT confirmed"):
            module.ingest(path, settings)
        assert client.count(module.COLLECTION).count == 1
        monkeypatch.setattr(module, "write_qdrant", write)
        assert module.ingest(path, settings)["count"] == 2
        assert module.ingest(path, settings)["count"] == 2
        assert client.count(module.COLLECTION).count == 2
        original_ids = [card["id"] for card in source]
        source[0] = make_card(source[0]["raw_fields"] | {"Final Price": "123.456789012345678"})
        assert [card["id"] for card in source] == original_ids
        path.write_text(json.dumps(source))
        assert module.ingest(path, settings)["count"] == 2
        with psycopg.connect(
            **settings.postgres,
            autocommit=True,
            row_factory=dict_row,
        ) as connection:
            module.verify(connection, client, source, vectors)
            client.upsert(
                module.COLLECTION,
                points=[
                    models.PointStruct(
                        id=source[0]["id"],
                        vector=vectors[1].tolist(),
                        payload=module.payload(source[0]),
                    )
                ],
            )
            with pytest.raises(ValueError, match="Stored vector differs"):
                module.verify(connection, client, source, vectors)
            module.write_qdrant(client, source, vectors, True)
            connection.execute("UPDATE public.products SET brand = 'Synthetic corruption'")
            with pytest.raises(ValueError, match="PostgreSQL values differ"):
                module.verify(connection, client, source, vectors)
            # Fewer input IDs must not silently delete the other product.
            with pytest.raises(ValueError, match="extra product IDs"):
                module.check_postgres(connection, {source[0]["id"]})
    finally:
        original_close()


def test_incompatible_existing_table_is_untouched(settings):
    with psycopg.connect(**settings.postgres, autocommit=True, row_factory=dict_row) as connection:
        connection.execute("CREATE TABLE public.products (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO public.products VALUES ('synthetic-existing')")
        with pytest.raises(ValueError, match="schema is incompatible"):
            module.check_postgres(connection, set())
        assert connection.execute("SELECT id FROM public.products").fetchone()["id"] == (
            "synthetic-existing"
        )
