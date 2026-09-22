from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from evaluation.profiles import PROFILES

JUDGMENT_FIELDS = [
    "query_id",
    "query",
    "paper_id",
    "title",
    "year",
    "venue",
    "abstract",
    "appeared_in_semantic_only",
    "appeared_in_fast",
    "appeared_in_quality_local",
    "appeared_in_full_bge_reference",
    "semantic_only_rank",
    "fast_rank",
    "quality_local_rank",
    "full_bge_reference_rank",
    "relevance",
    "notes",
]


def pool_top_results(
    candidate_sets: dict[str, Any],
    ranking_documents: dict[str, dict[str, Any]],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    query_rankings: dict[str, dict[str, dict[str, Any]]] = {}
    for profile, document in ranking_documents.items():
        query_rankings[profile] = {row["query_id"]: row for row in document["queries"]}

    rows: list[dict[str, Any]] = []
    for query_data in candidate_sets["queries"]:
        query_id = query_data["query_id"]
        candidates = {paper["id"]: paper for paper in query_data["papers"]}
        pooled_ids: list[str] = []
        seen: set[str] = set()
        ranks: dict[str, dict[str, int]] = {name: {} for name in PROFILES}

        for profile in PROFILES:
            ranking = query_rankings.get(profile, {}).get(query_id, {})
            for result in ranking.get("results", [])[:top_k]:
                paper_id = result["paper_id"]
                ranks[profile][paper_id] = result["rank"]
                if paper_id not in seen:
                    seen.add(paper_id)
                    pooled_ids.append(paper_id)

        for paper_id in pooled_ids:
            paper = candidates[paper_id]
            row: dict[str, Any] = {
                "query_id": query_id,
                "query": query_data["query"],
                "paper_id": paper_id,
                "title": paper["title"],
                "year": paper.get("publication_year") or "",
                "venue": paper.get("venue") or "",
                "abstract": paper.get("abstract") or "",
                "relevance": "",
                "notes": "",
            }
            for profile in PROFILES:
                rank = ranks[profile].get(paper_id)
                row[f"appeared_in_{profile}"] = int(rank is not None)
                row[f"{profile}_rank"] = rank if rank is not None else ""
            rows.append(row)
    return rows


def validate_relevance_labels(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], int]:
    labels: dict[tuple[str, str], int] = {}
    missing: list[tuple[str, str]] = []
    invalid: list[tuple[str, str, Any]] = []
    for row in rows:
        key = (str(row["query_id"]), str(row["paper_id"]))
        raw_value = row.get("relevance", "")
        if raw_value is None or str(raw_value).strip() == "":
            missing.append(key)
            continue
        try:
            value = int(str(raw_value).strip())
        except ValueError:
            invalid.append((*key, raw_value))
            continue
        if value not in {0, 1, 2, 3}:
            invalid.append((*key, raw_value))
            continue
        labels[key] = value
    if missing:
        raise ValueError(
            f"{len(missing)} relevance label(s) are missing; fill every relevance cell with 0-3"
        )
    if invalid:
        raise ValueError(f"{len(invalid)} relevance label(s) are invalid; use integers 0-3")
    return labels
