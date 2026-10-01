from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from app.models.paper import Paper
from app.services.graph_semantics import (
    ABSTRACT_COMPLETENESS,
    INSIGHTS_COMPLETENESS,
    K_NEIGHBORS,
    METADATA_COMPLETENESS,
    MIN_EDGE_SIMILARITY,
    PARSED_PDF_COMPLETENESS,
    GraphSemanticsService,
    format_graph_seed_audit,
    information_completeness,
    relevance_to_opacity,
    select_sparse_edges,
)


class MappingEncoder:
    def __init__(self, vectors: dict[str, list[float]], query_vector: list[float] | None = None):
        self.vectors = vectors
        self.query_vector = query_vector or [1.0, 0.0, 0.0, 0.0]
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences: str | list[str], **kwargs: object) -> np.ndarray:
        assert isinstance(sentences, list)
        self.calls.append((sentences, kwargs))
        encoded: list[list[float]] = []
        paper_only_batch = sentences[0].startswith("Title: ")
        for index, text in enumerate(sentences):
            if index == 0 and not paper_only_batch:
                encoded.append(self.query_vector)
            elif text in self.vectors:
                encoded.append(self.vectors[text])
            elif text.startswith("Title: "):
                title = text.splitlines()[0].removeprefix("Title: ")
                encoded.append(self.vectors[title])
            else:
                encoded.append([0.5, 0.5, 0.5, 0.5])
        return np.asarray(encoded, dtype=float)


class NeverEncoder:
    def encode(self, sentences: str | list[str], **_: object) -> np.ndarray:
        raise AssertionError(f"Encoder should not be called: {sentences}")


def _matrix(paper_ids: list[str], values: dict[tuple[str, str], float]) -> np.ndarray:
    index = {paper_id: position for position, paper_id in enumerate(paper_ids)}
    matrix = np.eye(len(paper_ids), dtype=float)
    for (first, second), value in values.items():
        matrix[index[first], index[second]] = value
        matrix[index[second], index[first]] = value
    return matrix


@pytest.mark.parametrize(
    ("paper", "parsed", "insights", "expected"),
    [
        (Paper(id="metadata", title="Metadata only"), False, False, METADATA_COMPLETENESS),
        (
            Paper(id="abstract", title="Abstract", abstract="Usable abstract."),
            False,
            False,
            ABSTRACT_COMPLETENESS,
        ),
        (Paper(id="parsed", title="Parsed"), True, False, PARSED_PDF_COMPLETENESS),
        (Paper(id="insights", title="Insights"), False, True, INSIGHTS_COMPLETENESS),
    ],
)
def test_information_completeness_states(
    paper: Paper, parsed: bool, insights: bool, expected: float
) -> None:
    completeness = information_completeness(
        paper,
        parsed_pdf_available=parsed,
        structured_insights_available=insights,
    )

    assert completeness == expected
    assert 0.0 <= completeness <= 1.0


def test_highest_valid_information_state_remains_available_but_relevance_drives_opacity() -> None:
    paper = Paper(id="paper", title="Complete paper", abstract="Available abstract")
    node = GraphSemanticsService("existing-model", NeverEncoder()).build_node_seeds(
        "Complete paper",
        [paper],
        parsed_paper_ids={paper.id},
        insight_paper_ids={paper.id},
    )[0]

    assert node.information_completeness == INSIGHTS_COMPLETENESS
    assert node.node_opacity == 1.0


def test_opacity_is_independent_of_metadata_at_equal_query_relevance() -> None:
    vectors = {"Same content": [0.8, 0.6, 0.0, 0.0]}
    papers = [
        Paper(
            id="low-metadata",
            title="Same content",
            abstract="Same abstract",
            citation_count=0,
            publication_year=2018,
            source_names=["openalex"],
        ),
        Paper(
            id="high-metadata",
            title="Same content",
            abstract="Same abstract",
            citation_count=50_000,
            publication_year=2026,
            source_names=["semantic_scholar", "arxiv"],
        ),
    ]
    nodes = GraphSemanticsService("existing-model", MappingEncoder(vectors)).build_node_seeds(
        "different query", papers
    )

    assert nodes[0].information_completeness == nodes[1].information_completeness == 0.55
    assert nodes[0].node_opacity == nodes[1].node_opacity
    assert nodes[0].node_opacity == pytest.approx(relevance_to_opacity(0.55))


def test_same_completeness_uses_same_opacity_at_different_query_relevance() -> None:
    paper = Paper(id="paper", title="Exact Paper Title", abstract="Available abstract")
    exact_node = GraphSemanticsService("existing-model", NeverEncoder()).build_node_seeds(
        "Exact Paper Title", [paper]
    )[0]
    weak_node = GraphSemanticsService(
        "existing-model",
        MappingEncoder({paper.title: [0.0, 1.0, 0.0, 0.0]}),
    ).build_node_seeds("unrelated query", [paper])[0]

    assert exact_node.query_relevance > weak_node.query_relevance
    assert exact_node.node_opacity == weak_node.node_opacity
    assert exact_node.node_opacity == pytest.approx(relevance_to_opacity(0.55))


def test_similarity_is_symmetric_bounded_semantic_and_title_fallback_is_safe() -> None:
    encoder = MappingEncoder(
        {
            "Wireless Agent Security": [1.0, 0.1, 0.0, 0.0],
            "Autonomous Network Defense": [0.95, 0.2, 0.0, 0.0],
            "Fern Taxonomy": [0.0, 0.0, 0.0, 1.0],
        }
    )
    papers = [
        Paper(id="a", title="Wireless Agent Security", abstract="Agents defend networks."),
        Paper(id="b", title="Autonomous Network Defense", abstract="Agentic cyber defense."),
        Paper(id="c", title="Fern Taxonomy", abstract=None),
    ]
    service = GraphSemanticsService("existing-model", encoder)

    similarities = service.paper_similarity_matrix(papers)

    assert similarities[0, 1] == pytest.approx(similarities[1, 0])
    assert similarities[0, 1] > similarities[0, 2]
    assert np.all((0.0 <= similarities) & (similarities <= 1.0))
    assert encoder.calls[0][0][2] == "Title: Fern Taxonomy"
    assert len(encoder.calls) == 1
    assert encoder.calls[0][1] == {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "batch_size": 16,
    }


def test_similarity_is_deterministic_and_ignores_query_scores_and_metadata() -> None:
    vectors = {
        "Paper A": [1.0, 0.0, 0.0, 0.0],
        "Paper B": [0.8, 0.6, 0.0, 0.0],
    }
    papers = [
        Paper(
            id="a",
            title="Paper A",
            abstract="Shared topic",
            semantic_score=0.99,
            reranker_score=12.0,
            citation_count=1_000,
            publication_year=2026,
            source_names=["openalex"],
        ),
        Paper(
            id="b",
            title="Paper B",
            abstract="Shared topic",
            semantic_score=0.01,
            reranker_score=-3.0,
            citation_count=0,
            publication_year=2017,
            source_names=["arxiv"],
        ),
    ]
    service = GraphSemanticsService("existing-model", MappingEncoder(vectors))

    first = service.paper_similarity_matrix(papers)
    second = service.paper_similarity_matrix(papers)

    assert np.array_equal(first, second)
    assert first[0, 1] == pytest.approx(0.8)


def test_edge_similarity_is_query_independent() -> None:
    papers = [Paper(id="a", title="Paper A"), Paper(id="b", title="Paper B")]
    vectors = {
        "Paper A": [1.0, 0.0, 0.0, 0.0],
        "Paper B": [0.8, 0.6, 0.0, 0.0],
    }
    first_service = GraphSemanticsService("existing-model", MappingEncoder(vectors))
    second_service = GraphSemanticsService("existing-model", MappingEncoder(vectors))

    first_service.build_node_seeds("first unrelated query", papers)
    second_service.build_node_seeds("second unrelated query", papers)

    assert first_service.build_edge_seeds(papers) == second_service.build_edge_seeds(papers)


def test_threshold_and_no_self_edges_allow_isolated_nodes() -> None:
    paper_ids = ["a", "b", "isolated"]
    similarities = _matrix(paper_ids, {("a", "b"): 0.34})

    edges = select_sparse_edges(paper_ids, similarities)

    assert edges == []
    assert all(edge.source_paper_id != edge.target_paper_id for edge in edges)


def test_union_selection_reason_and_canonical_no_duplicate_edges() -> None:
    paper_ids = ["a", "b", "c", "d", "e"]
    similarities = _matrix(
        paper_ids,
        {
            ("a", "b"): 0.90,
            ("a", "c"): 0.80,
            ("a", "d"): 0.70,
            ("a", "e"): 0.60,
            ("b", "c"): 0.95,
            ("b", "d"): 0.94,
            ("b", "e"): 0.93,
            ("c", "d"): 0.92,
            ("c", "e"): 0.91,
            ("d", "e"): 0.89,
        },
    )

    edges = select_sparse_edges(paper_ids, similarities)
    pairs = [(edge.source_paper_id, edge.target_paper_id) for edge in edges]

    assert ("a", "e") not in pairs
    assert len(pairs) == len(set(pairs))
    assert all(source < target for source, target in pairs)
    assert any(edge.selection_reason == "source_top_k" for edge in edges)
    assert any(edge.selection_reason == "both_top_k" for edge in edges)


def test_one_sided_top_k_selection_is_enough_for_union_edge() -> None:
    paper_ids = ["a", "b", "c", "d", "e"]
    similarities = _matrix(
        paper_ids,
        {
            ("a", "b"): 0.90,
            ("a", "c"): 0.80,
            ("a", "d"): 0.70,
            ("a", "e"): 0.60,
        },
    )

    edges = select_sparse_edges(paper_ids, similarities)
    edge = next(
        edge for edge in edges if (edge.source_paper_id, edge.target_paper_id) == ("a", "e")
    )

    assert edge.selection_reason == "target_top_k"
    assert edge.paper_similarity == 0.60


def test_top_k_uses_deterministic_id_tie_breaking() -> None:
    paper_ids = ["a", "b", "c", "d", "e"]
    similarities = _matrix(
        paper_ids,
        {
            ("a", "b"): 0.80,
            ("a", "c"): 0.80,
            ("a", "d"): 0.80,
            ("a", "e"): 0.80,
            ("b", "e"): 0.95,
            ("c", "e"): 0.94,
            ("d", "e"): 0.93,
        },
    )

    edges = select_sparse_edges(paper_ids, similarities)
    by_pair = {(edge.source_paper_id, edge.target_paper_id): edge for edge in edges}

    assert ("a", "b") in by_pair
    assert ("a", "c") in by_pair
    assert ("a", "d") in by_pair
    assert ("a", "e") not in by_pair


def test_stronger_similarity_has_larger_equal_weight_and_graph_is_sparse() -> None:
    paper_ids = ["a", "b", "c", "d"]
    similarities = _matrix(
        paper_ids,
        {("a", "b"): 0.91, ("a", "c"): 0.61, ("b", "c"): 0.40},
    )

    edges = select_sparse_edges(paper_ids, similarities)
    weights = {(edge.source_paper_id, edge.target_paper_id): edge.edge_weight for edge in edges}

    assert weights[("a", "b")] > weights[("a", "c")]
    assert all(edge.edge_weight == edge.paper_similarity for edge in edges)
    assert len(edges) < len(paper_ids) * (len(paper_ids) - 1) // 2


def test_realistic_graph_fixture_is_sparse_separates_semantics_and_keeps_outlier() -> None:
    vectors = {
        "Agentic Security for Wireless Networks": [1.0, 0.1, 0.0, 0.0],
        "Autonomous Defense in Mobile Networks": [0.95, 0.2, 0.05, 0.0],
        "Security Risks of Network Agents": [0.8, 0.3, 0.2, 0.0],
        "Visual Analytics for Complex Data": [0.1, 1.0, 0.1, 0.0],
        "Interactive Visualization Agents": [0.15, 0.9, 0.25, 0.0],
        "Fern Leaf Taxonomy": [0.0, 0.0, 0.0, 1.0],
        "Clinical Cohort Analysis": [0.1, 0.2, 1.0, 0.0],
    }
    papers = [
        Paper(id="wireless-a", title=title, abstract=f"Research about {title.lower()}.")
        for title in vectors
    ]
    papers[1].id = "wireless-b"
    papers[2].id = "wireless-c"
    papers[3].id = "visual-d"
    papers[4].id = "visual-e"
    papers[5].id = "botany-f"
    papers[5].abstract = None
    papers[6].id = "clinical-g"
    original_values = copy.deepcopy([paper.model_dump() for paper in papers])
    service = GraphSemanticsService(
        "sentence-transformers/all-MiniLM-L6-v2", MappingEncoder(vectors)
    )

    first = service.build_graph_seed(
        "agentic security in wireless networks",
        papers,
        parsed_paper_ids={"visual-d"},
        insight_paper_ids={"wireless-a"},
    )
    second = service.build_graph_seed(
        "agentic security in wireless networks",
        papers,
        parsed_paper_ids={"visual-d"},
        insight_paper_ids={"wireless-a"},
    )
    pairs = {(edge.source_paper_id, edge.target_paper_id) for edge in first.edges}
    connected_ids = {paper_id for pair in pairs for paper_id in pair}
    nodes = {node.paper_id: node for node in first.nodes}

    assert first == second
    assert ("wireless-a", "wireless-b") in pairs
    assert "botany-f" not in connected_ids
    assert len(first.edges) == 9
    assert len(first.edges) < len(papers) * (len(papers) - 1) // 2
    assert nodes["wireless-a"].node_radius > nodes["botany-f"].node_radius
    assert nodes["wireless-a"].node_opacity == pytest.approx(
        relevance_to_opacity(nodes["wireless-a"].information_completeness)
    )
    assert nodes["visual-d"].node_opacity == pytest.approx(
        relevance_to_opacity(nodes["visual-d"].information_completeness)
    )
    assert all(edge.edge_weight == edge.paper_similarity for edge in first.edges)
    assert [paper.model_dump() for paper in papers] == original_values
    audit = format_graph_seed_audit(first)
    assert "NODES:" in audit
    assert "EDGES:" in audit
    assert "information_completeness=" in audit
    assert "selection_reason=" in audit
    assert K_NEIGHBORS == 3
    assert MIN_EDGE_SIMILARITY == 0.35


def test_locally_cached_m1_model_builds_a_realistic_sparse_semantic_graph() -> None:
    snapshots_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    snapshots = sorted(path for path in snapshots_root.glob("*") if path.is_dir())
    if not snapshots:
        pytest.skip("The configured M1 embedding model is not already cached locally")

    from sentence_transformers import SentenceTransformer

    papers = [
        Paper(
            id="network-agent-a",
            title="Agentic AI Security in Wireless Networks",
            abstract="Autonomous language-model agents detect and mitigate attacks in 6G networks.",
        ),
        Paper(
            id="network-agent-b",
            title="LLM Agents for Secure Mobile Communications",
            abstract="Agent-based cyber defense protects wireless communication infrastructure.",
        ),
        Paper(
            id="network-agent-c",
            title="Autonomous Network Operations with AI Agents",
            abstract="Planning agents monitor and operate next-generation communication networks.",
        ),
        Paper(
            id="visualization-d",
            title="Interactive Visual Analytics for Agent Systems",
            abstract="Visual interfaces help analysts understand multi-agent behavior.",
        ),
        Paper(
            id="health-e",
            title="Cancer Care Disparities Across Clinical Centers",
            abstract="Clinical and socioeconomic data reveal differences between patient cohorts.",
        ),
        Paper(
            id="botany-f",
            title="A Taxonomy of Fern Leaf Morphology",
            abstract="Botanical specimens are classified using frond and spore characteristics.",
        ),
        Paper(
            id="biology-g",
            title="Protein Folding Dynamics in Molecular Biology",
            abstract="Molecular simulations characterize protein structure and folding pathways.",
        ),
    ]
    encoder = SentenceTransformer(str(snapshots[-1]))
    service = GraphSemanticsService("sentence-transformers/all-MiniLM-L6-v2", encoder)

    similarities = service.paper_similarity_matrix(papers)
    edges = service.build_edge_seeds(papers)
    pairs = {(edge.source_paper_id, edge.target_paper_id) for edge in edges}

    assert similarities[0, 1] > similarities[0, 5]
    assert ("network-agent-a", "network-agent-b") in pairs
    assert ("botany-f", "network-agent-a") not in pairs
    assert len(edges) < len(papers) * (len(papers) - 1) // 2
