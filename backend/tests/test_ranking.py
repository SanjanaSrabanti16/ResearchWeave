import numpy as np
import pytest

from app.models.paper import Paper
from app.services.ranking_service import (
    RankingService,
    is_exact_title_match,
    normalize_title_match_text,
    reciprocal_rank_fusion,
    rrf_score,
)


class FakeBiEncoder:
    def encode(self, sentences: str | list[str], **_: object) -> np.ndarray:
        if isinstance(sentences, str):
            return np.array([1.0, 0.0])
        return np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])[: len(sentences)]


class FakeReranker:
    def predict(self, sentences: list[list[str]], **_: object) -> np.ndarray:
        return np.array([0.1, 0.9, 0.4])[: len(sentences)]


def test_two_stage_ranking_keeps_both_scores_and_uses_reranker_order() -> None:
    papers = [Paper(title=name) for name in ["One", "Two", "Three"]]
    service = RankingService("fake-bi", "fake-cross", 3, FakeBiEncoder(), FakeReranker())
    ranked = service.rank("query", papers, 2)
    assert [paper.title for paper in ranked] == ["Three", "Two"]
    assert all(paper.semantic_score is not None for paper in ranked)
    assert all(paper.reranker_score is not None for paper in ranked)


def test_rrf_uses_rank_positions_and_breaks_ties_deterministically() -> None:
    one, two, three = [Paper(title=name) for name in ["One", "Two", "Three"]]
    assert rrf_score(1, 3) == pytest.approx(1 / 61 + 1 / 63)
    assert rrf_score(2, 2) == pytest.approx(2 / 62)
    assert reciprocal_rank_fusion([one, two, three], [three, two, one]) == [
        one,
        three,
        two,
    ]
    with pytest.raises(ValueError, match="positive"):
        rrf_score(1, 1, k=0)


def test_fast_rrf_order_is_switchable_and_preserves_model_scores() -> None:
    papers = [Paper(title=name) for name in ["One", "Two", "Three"]]
    service = RankingService(
        "fake-bi", "fake-cross", 3, FakeBiEncoder(), FakeReranker(), fusion_mode="rrf"
    )
    first = service.rank("query", papers, 3)
    second = service.rank("query", papers, 3)
    assert [paper.title for paper in first] == ["Three", "One", "Two"]
    assert [paper.title for paper in second] == [paper.title for paper in first]
    assert [(paper.semantic_score, paper.reranker_score) for paper in first] == [
        (paper.semantic_score, paper.reranker_score) for paper in second
    ]
    assert all(
        paper.semantic_score is not None and paper.reranker_score is not None for paper in first
    )


def test_exact_title_match_takes_priority_over_model_ranking() -> None:
    papers = [
        Paper(title="Attention Is All You Need"),
        Paper(title="Visual Attention"),
        Paper(title="Auditory Attention"),
    ]
    service = RankingService("fake-bi", "fake-cross", 3, FakeBiEncoder(), FakeReranker())

    ranked = service.rank("Attention Is All You Need", papers, 3)

    assert ranked[0].title == "Attention Is All You Need"
    assert ranked[0].semantic_score is not None
    assert ranked[0].reranker_score is not None


def test_exact_title_matching_normalizes_case_punctuation_and_whitespace() -> None:
    assert normalize_title_match_text("  ATTENTION: Is All You Need!!! ") == (
        "attention is all you need"
    )
    assert is_exact_title_match("  ATTENTION: Is All You Need!!! ", "Attention Is All You Need")


def test_partial_keyword_query_does_not_receive_exact_title_priority() -> None:
    papers = [
        Paper(title="Attention Is All You Need"),
        Paper(title="Visual Attention"),
        Paper(title="Auditory Attention"),
    ]
    service = RankingService("fake-bi", "fake-cross", 3, FakeBiEncoder(), FakeReranker())

    ranked = service.rank("Attention", papers, 3)

    assert [paper.title for paper in ranked] == [
        "Auditory Attention",
        "Visual Attention",
        "Attention Is All You Need",
    ]


def test_topical_query_keeps_existing_rrf_behavior() -> None:
    papers = [Paper(title=name) for name in ["One", "Two", "Three"]]
    service = RankingService(
        "fake-bi", "fake-cross", 3, FakeBiEncoder(), FakeReranker(), fusion_mode="rrf"
    )

    ranked = service.rank("AI agents for visualizations", papers, 3)

    assert [paper.title for paper in ranked] == ["Three", "One", "Two"]
