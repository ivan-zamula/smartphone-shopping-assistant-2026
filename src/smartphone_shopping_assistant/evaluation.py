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

from .contracts import SearchFilters, SearchRequest
from .ingest import COLLECTION, MODEL, REVISION
from .search import SearchEngine


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
        cases.append(
            Judgment(
                f"P{i:02}",
                "paraphrase",
                f"Show me {card['name']}",
                {},
                {card["id"]: 2},
                "Paraphrase retains the complete catalog name.",
                f"catalog.name={card['name']!r}; catalog.id={card['id']}",
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
    grades: dict[str, int], ranked: list[str], *, positive: bool, k: int = 5, n: int = 20,
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
) -> tuple[list[dict], dict]:
    """A later LLM ranker may reorder/subselect retrieved candidates, never add IDs."""
    rows = []
    for case in cases:
        request = SearchRequest(
            query=case.query, filters=SearchFilters.model_validate(case.filters)
        )
        start = time.perf_counter()
        try:
            candidates = engine.retrieve(request, candidate_limit=20)
            retrieval_ms = (time.perf_counter() - start) * 1000
            chosen = ranker(request, candidates) if ranker else candidates
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
                case.grades, ids, positive=bool(case.grades),
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
                    "total_ms": (time.perf_counter() - start) * 1000,
                    "violations": 0,
                }
            )
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
    for field in ("retrieval_ms", "total_ms"):
        summary[field + "_cold"] = rows[0][field] if rows else None
        warm = [r[field] for r in rows[1:]]
        summary[field + "_warm_p50"] = percentile(warm, 50)
        summary[field + "_warm_p95"] = percentile(warm, 95)
    summary["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("var/prepared/catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("var/evaluation"))
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
    rows, summary = evaluate(cases, {card["id"] for card in cards}, SearchEngine())
    lines = [
        "# Local retrieval baseline",
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
        "",
        (
            "Precision uses only judged results as denominator; unjudged hits are reported. "
            "Retrieval Recall@20 uses the original candidate list; selected Recall@20, "
            "precision and nDCG use the final list. Recall and nDCG use explicitly labeled "
            "positives only. Duplicate IDs contribute once to metrics and count as violations. "
            "Exact labels do not assert "
            "other products are negative. Unsupported attributes have no relevance labels. Cold is "
            "first request; warm p50/p95 use remaining requests. Stage timings beyond retrieval "
            "unavailable from SearchEngine."
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
    lines += [
        "",
        "## Scope",
        "",
        (
            "Retrieval baseline only. LLM ranking, dialogue, UI, unsupported-attribute handling, "
            "and A07/A11 end-to-end acceptance: NOT RUN. Per-stage SQL/embedding/Qdrant "
            "timings: NOT AVAILABLE. RSS is peak process RSS in KiB on Linux. "
            "Query labels and report are local; no user-facing claim "
            "follows from this small, partly judged set."
        ),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
