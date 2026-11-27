"""Local, reproducible retrieval evaluation against catalog-derived judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from .contracts import SearchFilters, SearchRequest
from .ingest import COLLECTION, MODEL, REVISION
from .llm import MODEL as LLM_MODEL
from .llm import SELECTION_NUM_CTX, LocalLLM
from .search import SearchEngine


def ollama_revision() -> str | None:
    """Return local model digest when Ollama exposes one."""
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2, trust_env=False)
        response.raise_for_status()
        for model in response.json().get("models", []):
            if model.get("name") == LLM_MODEL:
                return model.get("digest") or None
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    return None


@dataclass(frozen=True)
class Judgment:
    case_id: str
    category: str
    query: str
    filters: dict
    grades: dict[str, int]
    rationale: str
    provenance: str


def build_cases(cards: list[dict]) -> list[Judgment]:
    """Derive judgments from source fields only; never from search results."""
    cards = sorted(cards, key=lambda x: (x["name"].casefold(), x["id"]))
    if len({x["id"] for x in cards}) != len(cards):
        raise ValueError("Duplicate catalog IDs")
    names = {}
    for card in cards:
        names.setdefault(card["name"].casefold(), []).append(card)
    unique = [rows[0] for rows in names.values() if len(rows) == 1]
    if len(unique) < 16:
        raise ValueError("Catalog has too few unique full names")
    # Spread selections over the sorted source, without sampling or retrieval feedback.
    complete = [x for x in unique if x["ram_gb"] is not None and x["storage_gb"] is not None]
    if len(complete) < 16:
        raise ValueError("Catalog has too few complete unique names")
    selected = [complete[(i * len(complete)) // 16] for i in range(16)]
    cases = []
    for i, card in enumerate(selected[:8], 1):
        cases.append(
            Judgment(
                f"E{i:02}",
                "exact_model",
                card["name"],
                {},
                {card["id"]: 2},
                "Exact full catalog name; only this UUID is confirmed relevant.",
                f"catalog.name={card['name']!r}; catalog.id={card['id']}",
            )
        )
    for i, card in enumerate(selected[8:], 1):
        if card["ram_gb"] is None or card["storage_gb"] is None:
            raise ValueError("Selected combination lacks memory fields")
        filters = {
            "brands": [card["brand"]],
            "ram_gb": {"minimum": card["ram_gb"], "maximum": card["ram_gb"]},
            "storage_gb": {"minimum": card["storage_gb"], "maximum": card["storage_gb"]},
        }
        matching = [
            x
            for x in cards
            if x["brand"].casefold() == card["brand"].casefold()
            and x["ram_gb"] == card["ram_gb"]
            and x["storage_gb"] == card["storage_gb"]
        ]
        cases.append(
            Judgment(
                f"C{i:02}",
                "combination",
                f"{card['brand']} with {card['ram_gb']} GB RAM and {card['storage_gb']} GB storage",
                filters,
                {x["id"]: 1 for x in matching},
                (
                    "Every catalog row satisfying all three exact fields is relevant; "
                    "no other semantic preference is asserted."
                ),
                f"catalog.brand/ram_gb/storage_gb; {len(matching)} matching UUIDs",
            )
        )
    for i, card in enumerate(selected[:8], 1):
        brand, model = card["brand"], card["model"]
        ram, storage = card["ram_gb"], card["storage_gb"]
        variants = (
            (f"Ищу {brand} {model}, какие варианты есть?", ("brand", "model")),
            (f"Покажи телефоны {brand} с {ram} ГБ ОЗУ", ("brand", "ram_gb")),
            (f"Есть смартфоны {brand} на {storage} ГБ памяти?", ("brand", "storage_gb")),
            (
                f"Подбери {brand}: ОЗУ {ram} ГБ, накопитель {storage} ГБ",
                ("brand", "ram_gb", "storage_gb"),
            ),
            (f"Нужен {brand} {model} с {ram} ГБ оперативной памяти", ("brand", "model", "ram_gb")),
            (
                f"Какие {brand} {model} имеют {storage} ГБ встроенной памяти?",
                ("brand", "model", "storage_gb"),
            ),
            (
                f"Найди {brand} с памятью {storage} ГБ и ОЗУ {ram} ГБ",
                ("brand", "storage_gb", "ram_gb"),
            ),
            (f"Покажи модель {model} бренда {brand}", ("brand", "model")),
        )
        query, fields = variants[i - 1]
        matching = [
            x
            for x in cards
            if all(
                x[field].casefold() == card[field].casefold()
                if field in ("brand", "model")
                else x[field] == card[field]
                for field in fields
            )
        ]
        cases.append(
            Judgment(
                f"P{i:02}",
                "paraphrase",
                query,
                {},
                {x["id"]: 1 for x in matching},
                (
                    "All source rows matching the stated exact fields are relevant; "
                    "no unspoken preference or unique target is assumed."
                ),
                f"catalog {', '.join(fields)}; {len(matching)} matching UUIDs",
            )
        )
    for i, attribute in enumerate(("camera megapixels", "battery capacity", "water resistance"), 1):
        cases.append(
            Judgment(
                f"U{i:02}",
                "unsupported",
                f"Which phones have {attribute}?",
                {},
                {},
                (
                    f"{attribute} is absent from the structured catalog; "
                    "relevance is unjudged, not negative."
                ),
                "catalog schema: name, brand, model, RAM, storage, color, unlocked, price",
            )
        )
    impossible = [
        ("brand", "Find a phone by brand __NO_SUCH_BRAND__", {"brands": ["__NO_SUCH_BRAND__"]}),
        ("ram", "Find a phone with at least 999999 GB RAM", {"ram_gb": {"minimum": 999999}}),
        (
            "price",
            "Find a phone cheaper than every listed phone",
            {"price": {"maximum": str(min(float(x["price"]) for x in cards) / 2)}},
        ),
    ]
    for i, (kind, query, filters) in enumerate(impossible, 1):
        cases.append(
            Judgment(
                f"I{i:02}",
                "impossible",
                query,
                filters,
                {},
                "No catalog row satisfies the exact filter; expected empty result.",
                f"catalog-wide {kind} comparison",
            )
        )
    return cases


def score_ranking(
    grades: dict[str, int],
    ranked: list[str],
    *,
    positive: bool,
    k: int = 5,
    n: int = 20,
    expected_empty: bool = False,
) -> dict:
    """Metrics with explicit unjudged coverage; unknown IDs never become negatives."""
    if k < 1 or n < 1:
        raise ValueError("Cutoffs must be positive")
    # A product can contribute only once, even if a faulty ranker repeats it.
    ranked = list(dict.fromkeys(ranked))
    positives = {item for item, grade in grades.items() if grade > 0}
    top = ranked[:k]
    judged = [item for item in top if item in grades]

    def gain(grade):
        return 2**grade - 1

    dcg = sum(gain(grades.get(item, 0)) / math.log2(i + 2) for i, item in enumerate(top))
    ideal = sum(
        gain(grade) / math.log2(i + 2)
        for i, grade in enumerate(sorted(grades.values(), reverse=True)[:k])
    )
    return {
        "recall": len(positives.intersection(ranked[:n])) / len(positives) if positives else None,
        "precision_judged": sum(grades[item] > 0 for item in judged) / len(judged)
        if judged
        else None,
        "judged_at_k": len(judged),
        "unjudged_at_k": len(top) - len(judged),
        "ndcg": dcg / ideal if ideal else None,
        "false_empty": bool(positive and not ranked),
        "nonempty_impossible": bool(expected_empty and ranked),
    }


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def evaluate(
    cases: list[Judgment],
    catalog_ids: set[str],
    engine: SearchEngine,
    ranker: Callable[[SearchRequest, list], list] | None = None,
    progress: Callable[[dict], None] | None = None,
    collect_timings: bool = False,
) -> tuple[list[dict], dict]:
    """A later LLM ranker may reorder/subselect retrieved candidates, never add IDs."""
    rows = []
    for case in cases:
        request = SearchRequest(
            query=case.query, filters=SearchFilters.model_validate(case.filters)
        )
        start = time.perf_counter()
        stages = {}
        try:
            candidates = (
                engine.retrieve(request, candidate_limit=20, timings=stages)
                if collect_timings
                else engine.retrieve(request, candidate_limit=20)
            )
            retrieval_ms = (time.perf_counter() - start) * 1000
            llm_start = time.perf_counter()
            chosen = ranker(request, candidates) if ranker else candidates
            llm_ms = (time.perf_counter() - llm_start) * 1000 if ranker else None
            ids = [str(item.product.id) for item in chosen]
            retrieved_ids = [str(item.product.id) for item in candidates]
            allowed = {str(item.product.id) for item in candidates}
            violations = [
                item
                for item in chosen
                if str(item.product.id) not in catalog_ids
                or str(item.product.id) not in allowed
                or not engine._matches(item.product, request.filters)
            ]
            duplicates = len(ids) - len(set(ids))
            metrics = score_ranking(
                case.grades,
                ids,
                positive=bool(case.grades),
                expected_empty=case.category == "impossible",
            )
            retrieval_recall = score_ranking(
                case.grades, retrieved_ids, positive=bool(case.grades)
            )["recall"]
            rows.append(
                {
                    "id": case.case_id,
                    "category": case.category,
                    "ids": ids,
                    "retrieval_recall": retrieval_recall,
                    "retrieval_ms": retrieval_ms,
                    **stages,
                    "llm_ms": llm_ms,
                    "total_ms": (time.perf_counter() - start) * 1000,
                    "violations": len(violations) + duplicates,
                    "error": None,
                    **metrics,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": case.case_id,
                    "category": case.category,
                    "error": f"{type(exc).__name__}: {exc}",
                    "retrieval_ms": (time.perf_counter() - start) * 1000,
                    **stages,
                    "llm_ms": None,
                    "total_ms": (time.perf_counter() - start) * 1000,
                    "violations": 0,
                }
            )
        if progress is not None:
            progress(rows[-1])
    valid = [r for r in rows if not r["error"]]
    summary = {
        "cases": len(rows),
        "errors": len(rows) - len(valid),
        "violations": sum(r["violations"] for r in valid),
        "positive_cases": sum(bool(c.grades) for c in cases),
        "false_empties": sum(r["false_empty"] for r in valid),
        "nonempty_impossible": sum(
            r["nonempty_impossible"] for r in valid if r["category"] == "impossible"
        ),
    }
    for key in ("retrieval_recall", "recall", "precision_judged", "ndcg"):
        vals = [
            r[key]
            for r in valid
            if r[key] is not None and r["category"] not in ("unsupported", "impossible")
        ]
        name = "selected_recall" if key == "recall" else key
        summary[name] = statistics.mean(vals) if vals else None
        summary[name + "_denominator"] = len(vals)
    for field in (
        "retrieval_ms",
        "total_ms",
        "sql_ms",
        "embedding_ms",
        "qdrant_ms",
        "hydration_ms",
        "llm_ms",
    ):
        summary[field + "_cold"] = rows[0].get(field) if rows else None
        warm = [r[field] for r in rows[1:] if r.get(field) is not None]
        summary[field + "_warm_p50"] = percentile(warm, 50)
        summary[field + "_warm_p95"] = percentile(warm, 95)
    summary["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("var/prepared/catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("var/evaluation"))
    parser.add_argument("--llm", action="store_true", help="Compare baseline and local selection")
    args = parser.parse_args()
    raw = args.catalog.read_bytes()
    cards = json.loads(raw)
    cases = build_cases(cards)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_hash = hashlib.sha256(raw).hexdigest()
    dataset = {"catalog_sha256": catalog_hash, "cases": [case.__dict__ for case in cases]}
    (args.output_dir / "queries.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n"
    )
    engine = SearchEngine()
    catalog_ids = {card["id"] for card in cards}

    def progress(row):
        print(
            f"{row['id']} {'ERROR' if row['error'] else 'OK'} {row['total_ms']:.0f}ms", flush=True
        )

    rows, summary = evaluate(
        cases, catalog_ids, engine, progress=progress if args.llm else None, collect_timings=True
    )
    llm_rows = llm_summary = None
    if args.llm:
        local_llm = LocalLLM()
        llm_revision = ollama_revision()

        def select(request, candidates):
            if not candidates:
                return []
            pool = candidates[:10]
            choice = local_llm.select(request.query, pool, limit=5, deadline=time.monotonic() + 120)
            by_id = {item.product.id: item for item in pool}
            return [by_id[item_id] for item_id in choice.selected_ids]

        llm_rows, llm_summary = evaluate(
            cases, catalog_ids, engine, ranker=select, progress=progress, collect_timings=True
        )
    threshold = (
        llm_summary is not None
        and llm_summary["errors"] == 0
        and llm_summary["violations"] == 0
        and llm_summary["ndcg"] is not None
        and llm_summary["ndcg"] >= 0.8
    )
    no_regression = (
        llm_summary is not None
        and llm_summary["errors"] == 0
        and llm_summary["violations"] == 0
        and llm_summary["ndcg"] is not None
        and summary["ndcg"] is not None
        and llm_summary["ndcg"] >= summary["ndcg"]
    )
    lines = [
        "# Local retrieval evaluation" + (" and LLM selection" if args.llm else " baseline"),
        "",
        f"Catalog SHA-256: `{catalog_hash}`; cases: {len(cases)}.",
        (
            f"Encoder: `{MODEL}` revision `{REVISION}`; Qdrant collection: `{COLLECTION}`; "
            "candidate limit: 20; K=5; N=20."
        ),
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        *(
            [
                "",
                "## Local LLM selection",
                "",
                "```json",
                json.dumps(llm_summary, indent=2),
                "```",
                "",
                f"Model: `{LLM_MODEL}`; temperature=0, selection_num_ctx={SELECTION_NUM_CTX}, "
                "num_predict=384; "
                "C<=10, K<=5; selected_positions map to candidate UUIDs in code. "
                f"Ollama model digest: `{llm_revision or 'unavailable'}`. "
                "Same 30 source-derived labels and queries as baseline.",
                f"nDCG@5 >= 0.8: {'PASS' if threshold else 'FAIL'}; "
                f"no nDCG regression: {'PASS' if no_regression else 'FAIL'}.",
                "Unsupported attributes are unjudged here; dialogue handles clarification. "
                "Per-stage SQL/embedding/Qdrant/hydration/LLM timings are in aggregate JSON.",
            ]
            if args.llm
            else []
        ),
        "",
        (
            "Precision uses only judged results as denominator; unjudged hits are reported. "
            "Retrieval Recall@20 uses the original candidate list; selected Recall@20, "
            "precision and nDCG use the final list. Recall and nDCG use explicitly labeled "
            "positives only. Duplicate IDs contribute once to metrics and count as violations. "
            "Exact labels do not assert "
            "other products are negative. Unsupported attributes have no relevance labels. Cold is "
            "first request; warm p50/p95 use remaining available measurements."
        ),
        "",
        "## Cases",
        "",
        (
            "| ID | Category | Retrieval Recall@20 | Selected Recall@20 | "
            "Judged precision | nDCG | Unjudged@5 | Empty error | Violations | ms |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]

    def metric(row, key):
        value = row.get(key)
        return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    for row in rows:
        empty_error = row["error"] or (
            "yes" if row.get("false_empty") or row.get("nonempty_impossible") else "no"
        )
        cells = [
            row["id"],
            row["category"],
            metric(row, "retrieval_recall"),
            metric(row, "recall"),
            metric(row, "precision_judged"),
            metric(row, "ndcg"),
            metric(row, "unjudged_at_k"),
            empty_error,
            str(row["violations"]),
            f"{row['total_ms']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if llm_rows is not None:
        lines.extend(
            [
                "",
                "## Local selection cases",
                "",
                "| ID | Category | Selected Recall@20 | nDCG@5 | Error | ms |",
                "|---|---|---:|---:|---|---:|",
            ]
        )
        for row in llm_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["id"],
                        row["category"],
                        metric(row, "recall"),
                        metric(row, "ndcg"),
                        row["error"] or "—",
                        f"{row['total_ms']:.1f}",
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "## Scope",
        "",
        (
            (
                "Retrieval and local selection measured. "
                if args.llm
                else "Retrieval baseline only. LLM ranking not run. "
            )
            + "Dialogue, UI, unsupported-attribute handling, "
            "and A07/A11 end-to-end acceptance: NOT RUN. Per-stage timings are measured "
            "where the stage executes. RSS is peak process RSS in KiB on Linux. "
            "Query labels and report are local; no user-facing claim "
            "follows from this small, partly judged set."
        ),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
