from __future__ import annotations

import math

import numpy as np
import pytest

from app.models.paper import Paper
from app.services.graph_semantics import (
    MAX_RADIUS,
    MIN_NODE_OPACITY,
    MIN_RADIUS,
    GraphSemanticsInputError,
    GraphSemanticsService,
    relevance_to_opacity,
    relevance_to_radius,
)


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences: str | list[str], **kwargs: object) -> np.ndarray:
        assert isinstance(sentences, list)
        self.calls.append((sentences, kwargs))
        vectors = []
        for text in sentences:
            normalized = text.casefold()
            if text == sentences[0]:
                vectors.append([1.0, 0.0])
            elif "agentic wireless security" in normalized:
                vectors.append([0.95, math.sqrt(1 - 0.95**2)])
            elif "unrelated botany" in normalized:
                vectors.append([0.10, math.sqrt(1 - 0.10**2)])
            else:
                vectors.append([0.60, 0.80])
        return np.asarray(vectors)


class ExtremeEncoder:
    def encode(self, sentences: str | list[str], **_: object) -> np.ndarray:
        assert isinstance(sentences, list)
        return np.asarray([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])[
            : len(sentences)
        ]


class PartialTitleEncoder:
    def encode(self, sentences: str | list[str], **_: object) -> np.ndarray:
        assert isinstance(sentences, list)
        vectors = [[1.0, 0.0]]
        for text in sentences[1:]:
            similarity = 0.35 if "streetweave" in text.casefold() else 0.46
            vectors.append([similarity, math.sqrt(1 - similarity**2)])
        return np.asarray(vectors)


class NeverEncoder:
    def encode(self, sentences: str | list[str], **_: object) -> np.ndarray:
        raise AssertionError(f"Exact title scoring should not call the encoder: {sentences}")


def test_exact_normalized_title_gets_maximum_relevance_and_radius() -> None:
    paper = Paper(id="exact", title="Attention Is All You Need")
    service = GraphSemanticsService("existing-model", NeverEncoder())

    node = service.build_node_seeds("  ATTENTION: Is All You Need!!! ", [paper])[0]

    assert node.query_relevance == 1.0
    assert node.node_weight == 1.0
    assert node.node_radius == MAX_RADIUS


def test_related_paper_scores_above_unrelated_and_has_larger_node() -> None:
    papers = [
        Paper(
            id="related",
            title="Agentic Wireless Security",
            abstract="Autonomous defenses for mobile networks.",
        ),
        Paper(
            id="unrelated",
            title="Unrelated Botany",
            abstract="Plant taxonomy and leaf morphology.",
        ),
    ]
    nodes = GraphSemanticsService("existing-model", RecordingEncoder()).build_node_seeds(
        "security agents for wireless networks", papers
    )

    assert nodes[0].query_relevance > nodes[1].query_relevance
    assert nodes[0].node_radius > nodes[1].node_radius


def test_relevance_and_radius_are_bounded_with_fixed_area_mapping() -> None:
    papers = [Paper(id="high", title="High"), Paper(id="low", title="Low")]
    nodes = GraphSemanticsService("existing-model", ExtremeEncoder()).build_node_seeds(
        "bounded query", papers
    )

    assert [node.query_relevance for node in nodes] == [1.0, 0.0]
    assert [node.node_radius for node in nodes] == [MAX_RADIUS, MIN_RADIUS]
    assert [node.node_opacity for node in nodes] == [0.58, 0.58]
    assert relevance_to_radius(0.25) == pytest.approx(
        math.sqrt(MIN_RADIUS**2 + 0.25 * (MAX_RADIUS**2 - MIN_RADIUS**2))
    )
    assert all(0.0 <= node.query_relevance <= 1.0 for node in nodes)
    assert all(MIN_RADIUS <= node.node_radius <= MAX_RADIUS for node in nodes)
    assert relevance_to_opacity(0.5) == pytest.approx(0.65)
    assert all(MIN_NODE_OPACITY <= node.node_opacity <= 1.0 for node in nodes)


def test_repeated_scoring_is_deterministic_and_missing_abstract_uses_title() -> None:
    paper = Paper(id="title-only", title="Agentic Wireless Security", abstract=None)
    service = GraphSemanticsService("existing-model", RecordingEncoder())

    first = service.build_node_seeds("wireless agent security", [paper])
    second = service.build_node_seeds("wireless agent security", [paper])

    assert first == second
    assert 0.0 <= first[0].query_relevance <= 1.0
    assert first[0].title == paper.title


def test_original_query_is_the_only_query_anchor_and_embeddings_are_batched() -> None:
    encoder = RecordingEncoder()
    papers = [Paper(id="one", title="One"), Paper(id="two", title="Two")]
    generated_variants = ["expanded variant one", "expanded variant two"]

    GraphSemanticsService("existing-model", encoder).build_node_seeds("original user query", papers)

    assert len(encoder.calls) == 1
    encoded_inputs, kwargs = encoder.calls[0]
    assert encoded_inputs[0] == "original user query"
    assert all(variant not in encoded_inputs for variant in generated_variants)
    assert len(encoded_inputs) == 5
    assert encoded_inputs[1:3] == ["One", "Two"]
    assert encoded_inputs[3:] == ["Title: One", "Title: Two"]
    assert kwargs == {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "batch_size": 16,
    }


def test_distinctive_partial_title_floor_outranks_generic_semantic_match() -> None:
    target = Paper(
        id="streetweave",
        title=(
            "StreetWeave: A Declarative Grammar for Street-Overlaid Visualization "
            "of Multivariate Data"
        ),
    )
    generic = Paper(
        id="generic",
        title="Declarative Grammar Systems",
        abstract="A generic discussion of declarative grammar and visual representations.",
    )

    nodes = GraphSemanticsService("existing-model", PartialTitleEncoder()).build_node_seeds(
        "streetweave declarative grammar", [target, generic]
    )

    assert nodes[0].query_relevance >= 0.90
    assert nodes[0].query_relevance > nodes[1].query_relevance


def test_generic_two_term_title_overlap_does_not_receive_partial_title_floor() -> None:
    paper = Paper(id="generic", title="A Declarative Grammar for Scientific Diagrams")

    node = GraphSemanticsService("existing-model", PartialTitleEncoder()).build_node_seeds(
        "declarative grammar", [paper]
    )[0]

    assert node.query_relevance == pytest.approx(0.46)


def test_citations_year_and_provider_do_not_influence_query_relevance() -> None:
    common = {
        "title": "Agentic Wireless Security",
        "abstract": "Autonomous defenses for mobile networks.",
    }
    papers = [
        Paper(
            id="one",
            **common,
            citation_count=0,
            publication_year=2019,
            source_names=["openalex"],
        ),
        Paper(
            id="two",
            **common,
            citation_count=100_000,
            publication_year=2026,
            source_names=["semantic_scholar", "arxiv"],
        ),
    ]
    nodes = GraphSemanticsService("existing-model", RecordingEncoder()).build_node_seeds(
        "wireless network defenses", papers
    )

    assert nodes[0].query_relevance == nodes[1].query_relevance
    assert nodes[0].node_radius == nodes[1].node_radius


def test_graph_scoring_preserves_paper_objects_and_m1_order() -> None:
    papers = [
        Paper(id="rank-2", title="Agentic Wireless Security", semantic_score=0.4, reranker_score=2),
        Paper(id="rank-1", title="Unrelated Botany", semantic_score=0.8, reranker_score=5),
    ]
    original_order = [paper.id for paper in papers]
    original_values = [paper.model_dump() for paper in papers]

    nodes = GraphSemanticsService("existing-model", RecordingEncoder()).build_node_seeds(
        "wireless security agents", papers
    )

    assert [paper.id for paper in papers] == original_order
    assert [paper.model_dump() for paper in papers] == original_values
    assert [node.paper_id for node in nodes] == original_order


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_original_query_fails_clearly(query: str) -> None:
    service = GraphSemanticsService("existing-model", RecordingEncoder())

    with pytest.raises(GraphSemanticsInputError, match="Original query must not be empty"):
        service.build_node_seeds(query, [Paper(title="Paper")])
