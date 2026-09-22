from __future__ import annotations

import math
from typing import Any

import pytest

from app.models.paper import PaperCandidate
from evaluation.benchmark import (
    candidate_set_fingerprint,
    collect_candidate_sets,
    rank_candidate_sets,
    render_evaluation_report,
)
from evaluation.metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from evaluation.pooling import pool_top_results, validate_relevance_labels
from evaluation.profiles import PROFILES, EvaluationProfile, resolve_profiles


def candidate_sets() -> dict[str, Any]:
    return {
        "top_k": 20,
        "queries": [
            {
                "query_id": "q1",
                "query": "visual agents",
                "papers": [
                    {
                        "id": "p1",
                        "title": "First",
                        "abstract": "First abstract",
                        "publication_year": 2024,
                        "venue": "VIS",
                    },
                    {
                        "id": "p2",
                        "title": "Second",
                        "abstract": "Second abstract",
                        "publication_year": 2023,
                        "venue": "TVCG",
                    },
                ],
            }
        ],
    }


def ranking_document(profile: str, paper_ids: list[str]) -> dict[str, Any]:
    papers = {"p1": "First", "p2": "Second"}
    return {
        "queries": [
            {
                "query_id": "q1",
                "results": [
                    {"paper_id": paper_id, "rank": rank, "title": papers[paper_id]}
                    for rank, paper_id in enumerate(paper_ids, 1)
                ],
            }
        ]
    }


def test_precision_at_k_uses_binary_relevance_threshold() -> None:
    assert precision_at_k([3, 2, 1, 0, 2], 5) == 0.6


def test_ndcg_at_k_uses_graded_relevance() -> None:
    assert ndcg_at_k([3, 2, 0], 3) == 1.0
    assert ndcg_at_k([0, 3, 2], 3) < 1.0


def test_reciprocal_rank_uses_first_relevance_two_or_higher() -> None:
    assert reciprocal_rank([0, 1, 2, 3]) == 1 / 3
    assert reciprocal_rank([0, 1]) == 0.0


def test_pooling_takes_union_and_deduplicates_papers() -> None:
    rankings = {
        "semantic_only": ranking_document("semantic_only", ["p1", "p2"]),
        "fast": ranking_document("fast", ["p2", "p1"]),
    }
    rows = pool_top_results(candidate_sets(), rankings)
    assert [row["paper_id"] for row in rows] == ["p1", "p2"]
    assert rows[0]["appeared_in_semantic_only"] == 1
    assert rows[0]["appeared_in_fast"] == 1
    assert rows[0]["semantic_only_rank"] == 1
    assert rows[0]["fast_rank"] == 2
    assert rows[0]["relevance"] == ""


def test_missing_relevance_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="relevance label.*missing"):
        validate_relevance_labels([{"query_id": "q1", "paper_id": "p1", "relevance": ""}])


def test_every_profile_receives_the_same_fixed_candidate_set() -> None:
    seen: dict[str, list[str]] = {}

    class FakeRanker:
        def __init__(self, profile: EvaluationProfile) -> None:
            self.profile = profile

        def initialize(self) -> float:
            return 0.0

        def rank(self, query: str, papers: list[Any], limit: int) -> tuple[list[Any], float]:
            del query
            seen[self.profile.name] = [paper.id for paper in papers]
            for paper in papers:
                paper.semantic_score = 0.5
            return papers[:limit], 0.01

        def resource_metadata(self) -> dict[str, Any]:
            return {"approximate_combined_model_weight_bytes": 1}

    fixed_candidates = candidate_sets()
    expected_fingerprint = candidate_set_fingerprint(fixed_candidates["queries"][0]["papers"])
    documents = rank_candidate_sets(
        fixed_candidates,
        ["semantic_only", "fast"],
        FakeRanker,
    )

    assert seen == {"semantic_only": ["p1", "p2"], "fast": ["p1", "p2"]}
    assert fixed_candidates["queries"][0]["papers"][0].get("semantic_score") is None
    assert {
        document["queries"][0]["candidate_set_fingerprint"] for document in documents.values()
    } == {expected_fingerprint}


def test_evaluation_profile_configuration() -> None:
    semantic, fast, quality, reference = resolve_profiles(list(PROFILES))
    assert semantic.reranker_model is None
    assert fast.shortlist_size == 50
    assert quality.embedding_model == "BAAI/bge-base-en-v1.5"
    assert quality.reranker_model == "cross-encoder/ms-marco-MiniLM-L12-v2"
    assert quality.shortlist_size == 100
    assert reference.evaluation_only is True
    assert reference.embedding_model == "BAAI/bge-m3"


def test_evaluation_report_formats_aggregate_results_without_declaring_winner() -> None:
    results = {
        "profiles": ["fast"],
        "aggregate": {
            "fast": {
                "mean_precision_at_5": 0.8,
                "mean_precision_at_10": 0.7,
                "mean_ndcg_at_10": 0.75,
                "mean_ndcg_at_20": 0.72,
                "mean_mrr": 1.0,
                "mean_ranking_duration_seconds": 2.5,
            }
        },
        "resources": {"fast": {"approximate_combined_model_weight_bytes": 150_000_000}},
    }
    report = render_evaluation_report(results)
    assert "| fast | 0.800 | 0.700 | 0.750 | 0.720 | 1.000 | 2.500s | 150.0 MB |" in report
    assert "No winner is declared automatically" in report
    assert not math.isnan(precision_at_k([2], 1))


async def test_collection_spaces_semantic_scholar_query_batches() -> None:
    class FakeProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def search(
            self,
            query: str,
            limit: int,
            start_year: int | None = None,
            end_year: int | None = None,
        ) -> list[PaperCandidate]:
            del limit, start_year, end_year
            if self.name != "semantic_scholar":
                return []
            return [
                PaperCandidate(
                    id=f"paper-{query}",
                    title=f"Paper for {query}",
                    publication_year=2024,
                    source_names=[self.name],
                )
            ]

    monotonic_values = iter([100.0, 100.25, 101.05])
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    benchmark = {
        "start_year": 2018,
        "end_year": 2026,
        "top_k": 20,
        "queries": [
            {"id": "q1", "query": "first query"},
            {"id": "q2", "query": "second query"},
        ],
    }

    result = await collect_candidate_sets(
        benchmark,
        [FakeProvider("openalex"), FakeProvider("semantic_scholar")],
        candidate_target=40,
        sleep=record_sleep,
        monotonic=lambda: next(monotonic_values),
    )

    assert len(result["queries"]) == 2
    assert sleeps == pytest.approx([0.8])
