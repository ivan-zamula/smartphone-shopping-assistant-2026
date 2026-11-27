"""Read-only PostgreSQL-filtered Qdrant retrieval."""

import os
import time
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from uuid import UUID

import numpy as np
import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient, models

from .contracts import ProductCard, SearchCandidate, SearchRequest
from .ingest import COLLECTION, DIMENSION, MODEL, MODEL_FILES, REVISION
from .ingest import Settings as IngestSettings


class SearchUnavailable(RuntimeError):
    """A safe stage-specific retrieval failure."""


@dataclass(frozen=True)
class Retrieval:
    candidates: list[SearchCandidate]
    sql_empty: bool = False


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError


@lru_cache(maxsize=1)
def _cached_encoder():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    path = snapshot_download(
        MODEL, revision=REVISION, local_files_only=True, allow_patterns=MODEL_FILES
    )
    return SentenceTransformer(path, device="cpu", local_files_only=True)


def _uuid(value):
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise ValueError("Invalid product ID")


class SearchEngine:
    def __init__(self, settings=None, encoder=None):
        self.settings = settings or IngestSettings.from_env()
        self.encoder = encoder

    def _connect(self):
        return psycopg.connect(**self.settings.postgres)

    def known_brands(self) -> list[str]:
        """Read current brand vocabulary from the same catalogue used for filtering."""
        try:
            with closing(self._connect()) as conn:
                return [
                    row[0]
                    for row in conn.execute(
                        "SELECT DISTINCT brand FROM public.products ORDER BY brand"
                    ).fetchall()
                ]
        except Exception as exc:
            raise SearchUnavailable("PostgreSQL brand lookup unavailable") from exc

    def _eligible(self, filters):
        clauses, params = [], []
        if filters.brands:
            clauses.append("lower(btrim(brand)) = ANY(%s)")
            params.append([b.strip().lower() for b in filters.brands])
        for col, bounds in (
            ("price", filters.price),
            ("ram_gb", filters.ram_gb),
            ("storage_gb", filters.storage_gb),
        ):
            if bounds:
                if bounds.minimum is not None:
                    clauses.append(f"{col} {'>=' if bounds.include_minimum else '>'} %s")
                    params.append(bounds.minimum)
                if bounds.maximum is not None:
                    clauses.append(f"{col} {'<=' if bounds.include_maximum else '<'} %s")
                    params.append(bounds.maximum)
        if filters.carrier_unlocked is not None:
            clauses.append("carrier_unlocked = %s")
            params.append(filters.carrier_unlocked)
        query = "SELECT id FROM public.products" + (
            " WHERE " + " AND ".join(clauses) if clauses else ""
        )
        with closing(self._connect()) as conn:
            return [_uuid(row[0]) for row in conn.execute(query, params).fetchall()]

    def get_products(self, ids):
        if not ids:
            return {}
        columns = (
            "id",
            "name",
            "brand",
            "model",
            "ram_gb",
            "storage_gb",
            "color",
            "carrier_unlocked",
            "price",
            "currency",
            "currency_status",
            "source_currency",
            "conversion_applied",
            "raw_fields",
            "text",
        )
        with closing(psycopg.connect(**self.settings.postgres, row_factory=dict_row)) as conn:
            rows = conn.execute(
                f"SELECT {', '.join(columns)} FROM public.products WHERE id = ANY(%s)", (list(ids),)
            ).fetchall()
        return {_uuid(row["id"]): ProductCard.model_validate(row) for row in rows}

    @staticmethod
    def _matches(product, filters):
        if filters.brands and product.brand.strip().lower() not in {
            brand.strip().lower() for brand in filters.brands
        }:
            return False
        for value, bounds in (
            (product.price, filters.price),
            (product.ram_gb, filters.ram_gb),
            (product.storage_gb, filters.storage_gb),
        ):
            if bounds is None:
                continue
            if value is None and (bounds.minimum is not None or bounds.maximum is not None):
                return False
            if bounds.minimum is not None and (
                value < bounds.minimum or (value == bounds.minimum and not bounds.include_minimum)
            ):
                return False
            if bounds.maximum is not None and (
                value > bounds.maximum or (value == bounds.maximum and not bounds.include_maximum)
            ):
                return False
        return (
            filters.carrier_unlocked is None or product.carrier_unlocked == filters.carrier_unlocked
        )

    def retrieve(
        self, request: SearchRequest, candidate_limit=20, *, timings: dict | None = None
    ) -> list[SearchCandidate]:
        return self.retrieve_with_status(request, candidate_limit, timings=timings).candidates

    def retrieve_with_status(
        self,
        request: SearchRequest,
        candidate_limit=20,
        *,
        timings: dict | None = None,
        deadline: float | None = None,
    ) -> Retrieval:
        _check_deadline(deadline)
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        started = time.perf_counter()
        try:
            ids = self._eligible(request.filters)
            eligible = set(ids)
            if len(eligible) != len(ids):
                raise ValueError("Duplicate SQL product ID")
        except Exception as exc:
            _check_deadline(deadline)
            raise SearchUnavailable("PostgreSQL filter stage unavailable") from exc
        if timings is not None:
            timings["sql_ms"] = (time.perf_counter() - started) * 1000
        _check_deadline(deadline)
        if not ids:
            return Retrieval([], sql_empty=True)
        started = time.perf_counter()
        try:
            encoder = self.encoder or _cached_encoder()
            _check_deadline(deadline)
            vector = np.asarray(
                encoder.encode("query: " + request.query, normalize_embeddings=True, device="cpu"),
                dtype=np.float32,
            )
            if (
                vector.shape != (DIMENSION,)
                or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)
            ):
                raise ValueError("Invalid query embedding")
        except Exception as exc:
            _check_deadline(deadline)
            raise SearchUnavailable("Query encoder unavailable") from exc
        _check_deadline(deadline)
        if timings is not None:
            timings["embedding_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        try:
            with closing(QdrantClient(url=self.settings.qdrant_url)) as client:
                config = client.get_collection(COLLECTION).config.params.vectors
                _check_deadline(deadline)
                if (
                    not isinstance(config, models.VectorParams)
                    or config.size != DIMENSION
                    or config.distance != models.Distance.COSINE
                ):
                    raise ValueError("Incompatible collection")
                hits = client.query_points(
                    collection_name=COLLECTION,
                    query=vector.tolist(),
                    query_filter=models.Filter(
                        must=[models.HasIdCondition(has_id=[str(i) for i in ids])]
                    ),
                    limit=min(candidate_limit, len(ids)),
                    with_payload=True,
                ).points
                _check_deadline(deadline)
            ordered, seen = [], set()
            for hit in hits:
                product_id = _uuid(hit.id)
                if product_id not in eligible or product_id in seen:
                    raise ValueError("Foreign or duplicate Qdrant ID")
                payload = hit.payload or {}
                if (
                    payload.get("embedding_model") != MODEL
                    or payload.get("embedding_revision") != REVISION
                ):
                    raise ValueError("Incompatible vector provenance")
                if not np.isfinite(hit.score):
                    raise ValueError("Invalid vector score")
                seen.add(product_id)
                ordered.append((product_id, hit.score))
            ordered.sort(key=lambda item: (-item[1], str(item[0])))
        except Exception as exc:
            _check_deadline(deadline)
            raise SearchUnavailable("Qdrant retrieval unavailable") from exc
        if timings is not None:
            timings["qdrant_ms"] = (time.perf_counter() - started) * 1000
        _check_deadline(deadline)
        if not ordered:
            return Retrieval([])
        started = time.perf_counter()
        try:
            products = self.get_products([product_id for product_id, _ in ordered])
            _check_deadline(deadline)
            if set(products) != seen:
                raise ValueError("Missing PostgreSQL product")
            if any(not self._matches(product, request.filters) for product in products.values()):
                raise ValueError("Hydrated product no longer matches filters")
            result = [
                SearchCandidate(product=products[product_id], vector_score=score)
                for product_id, score in ordered
            ]
            if timings is not None:
                timings["hydration_ms"] = (time.perf_counter() - started) * 1000
            return Retrieval(result)
        except Exception as exc:
            _check_deadline(deadline)
            raise SearchUnavailable("PostgreSQL hydration unavailable") from exc
