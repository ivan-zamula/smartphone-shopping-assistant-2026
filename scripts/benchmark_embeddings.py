"""Download a pinned local model and time a 100-card sample (no Qdrant writes)."""

import csv
import hashlib
import io
import json
import os
import platform
import resource
import statistics
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DATASET = "juanmerinobermejo/smartphones-price-dataset"
CSV_SHA256 = "e0077c3d7bbe46304870a87c9a14796f339a0425e1d4088efcbb683511fc3de1"
ROOT = Path(__file__).resolve().parents[1]


def log(message):
    print(message, flush=True)


def main():
    started = time.perf_counter()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    imports_seconds = time.perf_counter() - started
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    log("Loading dataset version 1; original values are preserved.")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber=1"
    with urlopen(url, timeout=60) as response:
        archive_bytes = response.read(5_000_001)
    if len(archive_bytes) > 5_000_000:
        raise ValueError("Unexpected dataset archive size")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        raw = archive.read("smartphones.csv")
    if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("Dataset differs from the inspected version")
    raw_path = ROOT / "var/raw/smartphones.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))

    # Anchor queries have known matches; add distractors from the CSV to total 100.
    anchors = ["Nothing Phone (2)", "Galaxy A51", "iPhone 12 Pro Max", "Realme C55"]
    sample = []
    for anchor in anchors:
        sample.append(
            next(row for row in rows if anchor.casefold() in row["Smartphone"].casefold())
        )
    for row in rows:
        if row not in sample:
            sample.append(row)
        if len(sample) == 100:
            break
    texts = []
    for row in sample:
        texts.append(
            f"passage: Название: {row['Smartphone']}. Бренд: {row['Brand']}. "
            f"Модель: {row['Model']}. RAM, ГБ: {row['RAM'] or 'неизвестно'}. "
            f"Накопитель, ГБ: {row['Storage'] or 'неизвестно'}. Цвет: {row['Color']}. "
            f"Без привязки к оператору: {row['Free']}. "
            f"Цена: {row['Final Price']}. Валюта проекта: USD, задана пользователем; "
            "валюта источника не подтверждена, пересчёт не выполнялся."
        )

    log(f"Downloading {MODEL}@{REVISION}; CPU only, 2 threads.")
    before_download = time.perf_counter()
    model_path = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "model.safetensors",
            "1_Pooling/*",
        ],
    )
    download_seconds = time.perf_counter() - before_download
    log(f"Download/cache resolution: {download_seconds:.2f}s. Loading model from disk.")
    before_load = time.perf_counter()
    model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    load_seconds = time.perf_counter() - before_load
    before_warmup = time.perf_counter()
    model.encode(texts[:1], normalize_embeddings=True, show_progress_bar=False)
    warmup_seconds = time.perf_counter() - before_warmup

    card_runs = []
    vectors = None
    for number in range(3):
        before = time.perf_counter()
        vectors = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        seconds = time.perf_counter() - before
        card_runs.append(seconds)
        log(f"100 cards, run {number + 1}: {seconds:.2f}s")
    assert vectors.shape == (100, 384)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)

    queries = [
        ("Найди Nothing Phone (2)", "Nothing Phone (2)"),
        ("Покажи Самсунг Галакси А51", "Galaxy A51"),
        ("Найди Айфон 12 Про Макс", "iPhone 12 Pro Max"),
        ("Найди Realme C55", "Realme C55"),
    ]
    query_results = []
    query_timings = []
    for query, expected in queries:
        runs = []
        for _ in range(5):
            before = time.perf_counter()
            vector = model.encode(
                ["query: " + query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            runs.append(time.perf_counter() - before)
        scores = vectors @ vector
        top = np.argsort(-scores)[:5]
        hits = [{"name": sample[i]["Smartphone"], "score": round(float(scores[i]), 6)} for i in top]
        query_timings.extend(runs)
        query_results.append(
            {
                "query": query,
                "expected_name_contains": expected,
                "hit_at_5": any(expected.casefold() in hit["name"].casefold() for hit in hits),
                "embedding_seconds": runs,
                "top_5": hits,
            }
        )
    report = {
        "measured_at": datetime.now(UTC).isoformat(),
        "scope": "100-card CPU sample; no vector DB or LLM; not a full quality evaluation",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "model": MODEL,
        "revision": REVISION,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_files_bytes": sum(
            p.stat().st_size for p in Path(model_path).rglob("*") if p.is_file()
        ),
        "dataset": DATASET,
        "dataset_version": 1,
        "source_sha256": CSV_SHA256,
        "dataset_rows": len(rows),
        "sample_rows": len(sample),
        "sample_names": [row["Smartphone"] for row in sample],
        "dimensions": 384,
        "batch_size": 16,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "normalized": True,
        "imports_seconds": imports_seconds,
        "download_seconds": download_seconds,
        "load_from_disk_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "cards_100_seconds": card_runs,
        "cards_per_second": 100 / statistics.median(card_runs),
        "full_catalog_estimate_seconds": len(rows) / 100 * statistics.median(card_runs),
        "estimate_note": "Linear estimate only, full catalog has not been embedded",
        "single_query_median_seconds": statistics.median(query_timings),
        "single_query_p95_seconds": float(np.percentile(query_timings, 95)),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "queries": query_results,
    }
    output = ROOT / "var/benchmarks/embedding-sample.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    log(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "parameters",
                    "model_files_bytes",
                    "load_from_disk_seconds",
                    "cards_100_seconds",
                    "single_query_median_seconds",
                    "single_query_p95_seconds",
                    "peak_process_rss_mib",
                    "full_catalog_estimate_seconds",
                ]
            },
            ensure_ascii=False,
        )
    )
    log(f"Anchor hits@5: {sum(item['hit_at_5'] for item in query_results)}/{len(queries)}")
    log(f"Report: {output}")


if __name__ == "__main__":
    main()
