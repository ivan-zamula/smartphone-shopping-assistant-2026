"""Synthetic catalog tests; these do not require live databases or the model."""

import json
from contextlib import closing
from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from smartphone_shopping_assistant import ingest as module
from smartphone_shopping_assistant.prepare import make_card


def synthetic_card(name="Synthetic Phone 8/256GB", price="231.60"):
    return make_card(
        {
            "Smartphone": name,
            "Brand": "Synthetic",
            "Model": "Phone",
            "RAM": "8",
            "Storage": "256",
            "Color": "Blue",
            "Free": "Yes",
            "Final Price": price,
        }
    )


@pytest.fixture
def catalog_file(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([synthetic_card()]))
    return path


def test_read_cards_preserves_uuid_raw_fields_and_exact_decimal(catalog_file):
    card = synthetic_card(price="0.100000000000000001")
    catalog_file.write_text(json.dumps([card]))
    actual = module.read_cards(catalog_file)
    assert actual == [card]
    values = dict(zip(module.COLUMN_TYPES, module.row_values(actual[0]), strict=True))
    assert str(values["id"]) == card["id"]
    assert values["price"] == Decimal("0.100000000000000001")
    assert values["raw_fields"].obj == card["raw_fields"]


@pytest.mark.parametrize("data", [[], {}, [None], [True], [{"raw_fields": {}}]])
def test_invalid_catalog_shapes(catalog_file, data):
    catalog_file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.read_cards(catalog_file)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "invalid"),
        ("ram_gb", 8.0),
        ("storage_gb", True),
        ("text", "changed"),
        ("currency", "EUR"),
        ("price", 1.0),
        ("conversion_applied", True),
    ],
)
def test_card_fields_must_agree_with_raw_input(catalog_file, field, value):
    card = synthetic_card()
    card[field] = value
    catalog_file.write_text(json.dumps([card]))
    with pytest.raises(ValueError, match="prepared format"):
        module.read_cards(catalog_file)


def test_duplicate_ids_rejected(catalog_file):
    catalog_file.write_text(json.dumps([synthetic_card(), synthetic_card()]))
    with pytest.raises(ValueError, match="duplicate UUID"):
        module.read_cards(catalog_file)


def test_invalid_input_never_opens_database(catalog_file, monkeypatch):
    catalog_file.write_text("[]")
    connect = Mock()
    monkeypatch.setattr(module.psycopg, "connect", connect)
    with pytest.raises(ValueError):
        module.ingest(catalog_file, module.Settings({}, "http://127.0.0.1:6333"))
    connect.assert_not_called()


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((1, 3)),
        np.zeros((1, 384)),
        np.ones((1, 384)),
        np.full((1, 384), np.nan),
        np.full((1, 384), np.inf),
        np.ones((2, 384)),
    ],
)
def test_invalid_vectors_rejected(vectors):
    with pytest.raises(ValueError):
        module.validate_vectors(vectors, 1)


def test_settings_environment_overrides_dotenv(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('POSTGRES_PASSWORD="synthetic-only"\nPOSTGRES_PORT=5432\n')
    monkeypatch.setenv("POSTGRES_PORT", "55432")
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    settings = module.Settings.from_env(path)
    assert settings.postgres["port"] == 55432
    assert settings.postgres["password"] == "synthetic-only"
    assert "synthetic-only" not in repr(settings)


def test_missing_password_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        module.Settings.from_env(tmp_path / "absent")


@pytest.mark.parametrize("size,distance", [(3, models.Distance.COSINE), (384, models.Distance.DOT)])
def test_incompatible_collection_is_not_recreated(size, distance):
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            module.COLLECTION,
            vectors_config=models.VectorParams(
                size=size,
                distance=distance,
            ),
        )
        with pytest.raises(ValueError, match="384-dimensional Cosine"):
            module.check_qdrant(client, set())
        assert client.get_collection(module.COLLECTION).config.params.vectors.size == size


def test_extra_point_ids_are_not_deleted():
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            module.COLLECTION,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )
        vector = [1.0] + [0.0] * 383
        client.upsert(module.COLLECTION, points=[models.PointStruct(id=1, vector=vector)])
        with pytest.raises(ValueError, match="extra product IDs"):
            module.check_qdrant(client, set())
        assert client.count(module.COLLECTION).count == 1


def test_embedding_failure_happens_before_database_writes(catalog_file, monkeypatch):
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(module.psycopg, "connect", Mock(return_value=connection))
    monkeypatch.setattr(module, "QdrantClient", Mock(return_value=Mock()))
    monkeypatch.setattr(module, "check_postgres", Mock())
    monkeypatch.setattr(module, "check_qdrant", Mock(return_value=False))
    monkeypatch.setattr(module, "encode_cards", Mock(side_effect=ValueError("bad model")))
    write = Mock()
    monkeypatch.setattr(module, "write_postgres", write)
    with pytest.raises(RuntimeError, match="embeddings.*not committed"):
        module.ingest(catalog_file, module.Settings({}, "http://127.0.0.1:6333"))
    write.assert_not_called()


def test_incompatible_model_revision_is_not_overwritten():
    card = synthetic_card()
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            module.COLLECTION,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )
        meta = module.payload(card) | {"embedding_revision": "synthetic-other-revision"}
        client.upsert(
            module.COLLECTION,
            points=[
                models.PointStruct(
                    id=card["id"],
                    vector=[1.0] + [0.0] * 383,
                    payload=meta,
                )
            ],
        )
        with pytest.raises(ValueError, match="model/revision differs"):
            module.check_qdrant(client, {card["id"]})
        assert client.retrieve(module.COLLECTION, ids=[card["id"]])[0].payload == meta
