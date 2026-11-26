"""Load validated local catalog cards into PostgreSQL and Qdrant."""

import argparse
import json
import os
import time
from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from qdrant_client import QdrantClient, models

from smartphone_shopping_assistant.prepare import COLUMNS, make_card

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILES = ["*.json", "*.txt", "*.model", "model.safetensors", "1_Pooling/*"]
DIMENSION = 384
COLLECTION = "smartphones"
BATCH_SIZE = 128
# Column order is also the fixed, trusted source for SQL identifiers.
COLUMN_TYPES = {
    "id": "uuid",
    "name": "text",
    "brand": "text",
    "model": "text",
    "ram_gb": "integer",
    "storage_gb": "integer",
    "color": "text",
    "carrier_unlocked": "boolean",
    "price": "numeric",
    "currency": "text",
    "currency_status": "text",
    "source_currency": "text",
    "conversion_applied": "boolean",
    "raw_fields": "jsonb",
    "text": "text",
}
COLUMNS_SQL = ", ".join(COLUMN_TYPES)
UPSERT = (
    f"INSERT INTO public.products ({COLUMNS_SQL}) "
    f"VALUES ({', '.join(['%s'] * len(COLUMN_TYPES))}) ON CONFLICT (id) DO UPDATE SET "
    + ", ".join(f"{column} = EXCLUDED.{column}" for column in COLUMN_TYPES if column != "id")
)


class IngestionValidationError(ValueError):
    """A safe, actionable catalog/configuration error for CLI output."""


def log(message):
    print(message, flush=True)


@dataclass
class Settings:
    postgres: dict = field(repr=False)
    qdrant_url: str

    @classmethod
    def from_env(cls, path=Path(".env")):
        values = {**dotenv_values(path), **os.environ}
        password = values.get("POSTGRES_PASSWORD")
        if not password:
            raise IngestionValidationError("Set POSTGRES_PASSWORD in .env or the environment")
        return cls(
            postgres={
                "host": values.get("POSTGRES_HOST", "127.0.0.1"),
                "port": int(values.get("POSTGRES_PORT", "5432")),
                "dbname": values.get("POSTGRES_DB", "smartphones"),
                "user": values.get("POSTGRES_USER", "smartphones"),
                "password": password,
                "connect_timeout": 10,
            },
            qdrant_url=values.get("QDRANT_URL", "http://127.0.0.1:6333"),
        )


def read_cards(path):
    cards = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cards, list) or not cards:
        raise IngestionValidationError("Catalog must be a nonempty JSON array")
    ids = set()
    for number, card in enumerate(cards, start=1):
        if not isinstance(card, dict):
            raise IngestionValidationError(f"Card {number}: expected an object")
        raw = card.get("raw_fields")
        if not isinstance(raw, dict) or set(raw) != set(COLUMNS):
            raise IngestionValidationError(f"Card {number}: invalid raw_fields")
        if not all(isinstance(value, str) for value in raw.values()):
            raise IngestionValidationError(f"Card {number}: raw_fields must contain strings")
        expected = make_card(raw)
        if set(card) != set(expected) or any(
            type(card[key]) is not type(value) or card[key] != value
            for key, value in expected.items()
        ):
            raise IngestionValidationError(
                f"Card {number}: fields disagree with the prepared format"
            )
        if card["id"] in ids:
            raise IngestionValidationError(f"Card {number}: duplicate UUID")
        ids.add(card["id"])
    return cards


def row_values(card):
    values = dict(card)
    values["id"] = UUID(card["id"])
    values["price"] = Decimal(card["raw_fields"]["Final Price"])
    values["raw_fields"] = Jsonb(card["raw_fields"])
    return tuple(values[column] for column in COLUMN_TYPES)


def check_postgres(connection, expected_ids):
    exists = connection.execute("SELECT to_regclass('public.products') AS name").fetchone()
    if exists["name"] is None:
        return
    columns = connection.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'products'"
    ).fetchall()
    if {r["column_name"]: r["data_type"] for r in columns} != COLUMN_TYPES:
        raise IngestionValidationError(
            "Existing public.products schema is incompatible; no automatic migration"
        )
    primary = connection.execute(
        "SELECT a.attname FROM pg_index i "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
        "WHERE i.indrelid = 'public.products'::regclass AND i.indisprimary"
    ).fetchall()
    if [r["attname"] for r in primary] != ["id"]:
        raise IngestionValidationError(
            "Existing public.products must have UUID id as its primary key"
        )
    actual = {str(r["id"]) for r in connection.execute("SELECT id FROM public.products")}
    if actual - expected_ids:
        raise IngestionValidationError(
            "PostgreSQL contains extra product IDs; automatic deletion is disabled"
        )


def qdrant_records(client, with_vectors=False):
    offset = None
    while True:
        records, offset = client.scroll(
            COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        yield from records
        if offset is None:
            break


def payload(card):
    return {"text": card["text"], "embedding_model": MODEL, "embedding_revision": REVISION}


def check_qdrant(client, expected_ids):
    if not client.collection_exists(COLLECTION):
        return False
    config = client.get_collection(COLLECTION).config.params.vectors
    if not isinstance(config, models.VectorParams) or (
        config.size != DIMENSION or config.distance != models.Distance.COSINE
    ):
        raise IngestionValidationError(
            "Existing smartphones collection must use 384-dimensional Cosine vectors"
        )
    for record in qdrant_records(client):
        if str(record.id) not in expected_ids:
            raise IngestionValidationError(
                "Qdrant contains extra product IDs; automatic deletion is disabled"
            )
        meta = record.payload or {}
        if meta.get("embedding_model") != MODEL or meta.get("embedding_revision") != REVISION:
            raise IngestionValidationError(
                "Existing vector model/revision differs; collection was not changed"
            )
    return True


def validate_vectors(vectors, count):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (count, DIMENSION) or not np.isfinite(vectors).all():
        raise IngestionValidationError(
            "Embeddings must be finite vectors of shape (card count, 384)"
        )
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise IngestionValidationError("Embeddings must have unit norm")
    return vectors


def encode_cards(cards):
    # Use the already downloaded revision; ingestion must not download model files.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    model_path = snapshot_download(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        allow_patterns=MODEL_FILES,
    )
    model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    vectors = model.encode(
        ["passage: " + card["text"] for card in cards],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return validate_vectors(vectors, len(cards))


def write_postgres(connection, cards):
    with connection.transaction():
        connection.execute(Path(__file__).with_name("products.sql").read_text())
        with connection.cursor() as cursor:
            cursor.executemany(UPSERT, [row_values(card) for card in cards])


def write_qdrant(client, cards, vectors, exists):
    if not exists:
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )
    for start in range(0, len(cards), BATCH_SIZE):
        client.upsert(
            COLLECTION,
            wait=True,
            points=[
                models.PointStruct(id=card["id"], vector=vectors[i].tolist(), payload=payload(card))
                for i, card in enumerate(cards[start : start + BATCH_SIZE], start=start)
            ],
        )


def verify(connection, client, cards, vectors):
    expected = {card["id"]: card for card in cards}
    rows = connection.execute(f"SELECT {COLUMNS_SQL} FROM public.products").fetchall()
    if {str(row["id"]) for row in rows} != set(expected) or len(rows) != len(cards):
        raise IngestionValidationError("PostgreSQL ID set/count differs from the input")
    for row in rows:
        card = expected[str(row["id"])].copy()
        card["id"] = UUID(card["id"])
        card["price"] = Decimal(card["raw_fields"]["Final Price"])
        if row != card:
            raise IngestionValidationError("PostgreSQL values differ from the input")
    positions = {card["id"]: i for i, card in enumerate(cards)}
    seen = set()
    for record in qdrant_records(client, with_vectors=True):
        key = str(record.id)
        if key not in expected or key in seen:
            raise IngestionValidationError("Unexpected or duplicate Qdrant ID")
        if record.payload != payload(expected[key]):
            raise IngestionValidationError("Qdrant payload differs from the input/model revision")
        actual = validate_vectors([record.vector], 1)[0]
        if not np.allclose(actual, vectors[positions[key]], atol=1e-5):
            raise IngestionValidationError("Stored vector differs from the embedding for its UUID")
        seen.add(key)
    if seen != set(expected) or client.count(COLLECTION, exact=True).count != len(cards):
        raise IngestionValidationError("Qdrant ID set/count differs from the input")
    hits = client.query_points(
        COLLECTION,
        query=vectors[0].tolist(),
        limit=1,
        search_params=models.SearchParams(exact=True),
    ).points
    if not hits or str(hits[0].id) not in expected or hits[0].score < 0.9999:
        raise IngestionValidationError("Vector search smoke check failed")


def ingest(path, settings):
    started = time.perf_counter()
    cards = read_cards(path)
    ids = {card["id"] for card in cards}
    log(f"Validated {len(cards)} local cards; checking PostgreSQL and Qdrant.")
    stage = "preflight (no catalog writes)"
    committed = False
    try:
        with (
            psycopg.connect(
                **settings.postgres,
                autocommit=True,
                row_factory=dict_row,
            ) as connection,
            closing(QdrantClient(url=settings.qdrant_url, timeout=30)) as client,
        ):
            check_postgres(connection, ids)
            exists = check_qdrant(client, ids)
            stage = "embeddings (no catalog writes)"
            log(f"Encoding {len(cards)} cards on CPU using the cached model revision.")
            before = time.perf_counter()
            vectors = validate_vectors(encode_cards(cards), len(cards))
            embedding_seconds = time.perf_counter() - before
            log(f"Embeddings validated: {len(cards)} x {DIMENSION}, {embedding_seconds:.2f}s.")
            stage = "PostgreSQL transaction"
            write_postgres(connection, cards)
            committed = True
            stage = "Qdrant upload and cross-store verification"
            log("PostgreSQL committed; upserting Qdrant points and verifying both stores.")
            write_qdrant(client, cards, vectors, exists)
            verify(connection, client, cards, vectors)
    except Exception as error:
        state = (
            "PostgreSQL committed; cross-store completion is NOT confirmed. Re-run to repair."
            if committed
            else "PostgreSQL transaction was not committed."
        )
        detail = f" {error}" if isinstance(error, IngestionValidationError) else ""
        raise RuntimeError(f"{stage} failed ({type(error).__name__}).{detail} {state}") from error
    seconds = time.perf_counter() - started
    log(f"Verified {len(cards)} products and vectors with matching UUIDs in {seconds:.2f}s.")
    return {"count": len(cards), "embedding_seconds": embedding_seconds, "total_seconds": seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("var/prepared/catalog.json"))
    args = parser.parse_args()
    try:
        ingest(args.input, Settings.from_env())
    except Exception as error:
        parser.exit(1, f"Ingestion failed: {error}\n")


if __name__ == "__main__":
    main()
