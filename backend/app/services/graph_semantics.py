from __future__ import annotations

import math
import re
import threading
from collections.abc import Sequence, Set
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.paper import Paper
from app.services.normalization import normalize_title
from app.services.ranking_service import BiEncoder, is_exact_title_match

MIN_RADIUS = 8.0
MAX_RADIUS = 28.0
MIN_NODE_OPACITY = 0.30
METADATA_COMPLETENESS = 0.40
ABSTRACT_COMPLETENESS = 0.55
PARSED_PDF_COMPLETENESS = 0.78
INSIGHTS_COMPLETENESS = 1.00
K_NEIGHBORS = 3
MIN_EDGE_SIMILARITY = 0.35
GRAPH_SEMANTICS_VERSION = "m3.4-m3.6-v4"
MAX_RELATED_CONCEPTS = 7
MAX_CONCEPT_CANDIDATES = 256

_CONCEPT_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "between",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "these",
        "this",
        "through",
        "to",
        "toward",
        "towards",
        "using",
        "via",
        "we",
        "were",
        "with",
    }
)
_GENERIC_CONCEPT_TERMS = frozenset(
    {
        "analysis",
        "approach",
        "data",
        "method",
        "methods",
        "paper",
        "research",
        "result",
        "results",
        "study",
    }
)
_CONCEPT_FRAGMENT_PATTERN = re.compile(r"[.!?;:\n()\[\]{}]+")
_CONCEPT_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)

_TITLE_STOPWORDS = frozenset(
    {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with"}
)
_GENERIC_TITLE_TERMS = frozenset(
    {
        "analysis",
        "approach",
        "data",
        "declarative",
        "framework",
        "grammar",
        "method",
        "model",
        "paper",
        "research",
        "study",
        "system",
        "systems",
        "using",
        "visualization",
    }
)
STRONG_TITLE_COVERAGE_FLOOR = 0.90

SelectionReason = Literal["source_top_k", "target_top_k", "both_top_k"]


class GraphSemanticsInputError(ValueError):
    """The graph relevance input cannot be scored meaningfully."""


class GraphSemanticsUnavailableError(RuntimeError):
    """The configured local semantic model could not score graph nodes."""


class GraphNodeSeed(BaseModel):
    """Provider-independent paper identity and node visual semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    query_relevance: float = Field(ge=0.0, le=1.0)
    node_weight: float = Field(ge=0.0, le=1.0)
    node_radius: float = Field(ge=MIN_RADIUS, le=MAX_RADIUS)
    information_completeness: float = Field(ge=0.0, le=1.0)
    node_opacity: float = Field(ge=MIN_NODE_OPACITY, le=1.0)

    @model_validator(mode="after")
    def enforce_visual_semantics(self) -> GraphNodeSeed:
        if self.node_weight != self.query_relevance:
            raise ValueError("node_weight must equal query_relevance")
        if not math.isclose(
            self.node_opacity,
            relevance_to_opacity(self.information_completeness),
            abs_tol=1e-12,
        ):
            raise ValueError("node_opacity must be derived from information_completeness")
        return self


class RelatedConcept(BaseModel):
    """A compact result-set concept ranked against the original query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=120)
    query_similarity: float = Field(ge=0.0, le=1.0)


class GraphEdgeSeed(BaseModel):
    """One canonical undirected semantic relationship between two papers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_paper_id: str = Field(min_length=1)
    target_paper_id: str = Field(min_length=1)
    paper_similarity: float = Field(ge=0.0, le=1.0)
    edge_weight: float = Field(ge=0.0, le=1.0)
    selection_reason: SelectionReason

    @model_validator(mode="after")
    def enforce_edge_semantics(self) -> GraphEdgeSeed:
        if self.source_paper_id >= self.target_paper_id:
            raise ValueError("edge paper IDs must be distinct and in canonical order")
        if self.edge_weight != self.paper_similarity:
            raise ValueError("edge_weight must equal paper_similarity")
        return self


class GraphSeed(BaseModel):
    """Backend-only graph data seed; it contains no layout or UI state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantics_version: str = GRAPH_SEMANTICS_VERSION
    embedding_model: str = Field(min_length=1)
    nodes: list[GraphNodeSeed]
    edges: list[GraphEdgeSeed]
    related_concepts: list[RelatedConcept] = Field(default_factory=list)


def relevance_to_radius(relevance: float) -> float:
    """Map relevance to radius so displayed node area is linear in relevance."""
    if not math.isfinite(relevance):
        raise GraphSemanticsInputError("Query relevance must be finite")
    weight = min(1.0, max(0.0, float(relevance)))
    return math.sqrt(MIN_RADIUS**2 + weight * (MAX_RADIUS**2 - MIN_RADIUS**2))


def relevance_to_opacity(completeness: float) -> float:
    """Map existing information completeness to a visible, bounded opacity."""
    if not math.isfinite(completeness):
        raise GraphSemanticsInputError("Information completeness must be finite")
    weight = min(1.0, max(0.0, float(completeness)))
    return MIN_NODE_OPACITY + weight * (1.0 - MIN_NODE_OPACITY)


def _canonical_relevance(cosine_similarity: float) -> float:
    if not math.isfinite(cosine_similarity):
        raise GraphSemanticsUnavailableError("The embedding model returned a non-finite score")
    return min(1.0, max(0.0, float(cosine_similarity)))


def _paper_representation(paper: Paper) -> str:
    representation = f"Title: {paper.title}"
    if paper.abstract:
        representation += f"\nAbstract: {paper.abstract}"
    return representation


def _singular_concept_token(token: str) -> str:
    parts = token.casefold().split("-")
    normalized: list[str] = []
    for part in parts:
        if len(part) > 4 and part.endswith("ies"):
            part = f"{part[:-3]}y"
        elif len(part) > 3 and part.endswith("s") and not part.endswith(("is", "ss", "us")):
            part = part[:-1]
        normalized.append(part)
    return "-".join(normalized)


def _concept_key(tokens: Sequence[str]) -> str:
    return " ".join(_singular_concept_token(token) for token in tokens)


def _concept_runs(text: str) -> list[list[str]]:
    runs: list[list[str]] = []
    for fragment in _CONCEPT_FRAGMENT_PATTERN.split(text):
        current: list[str] = []
        for token in _CONCEPT_TOKEN_PATTERN.findall(fragment):
            if token.casefold() in _CONCEPT_STOPWORDS:
                if current:
                    runs.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            runs.append(current)
    return runs


def extract_concept_candidates(papers: Sequence[Paper]) -> list[str]:
    """Extract deterministic 1-3 word concept phrases from current result papers."""
    candidates: dict[str, tuple[str, int, int]] = {}
    for paper in papers:
        sources = ((paper.title, 4), (paper.abstract or "", 1))
        for text, source_weight in sources:
            for run in _concept_runs(text):
                for width in (3, 2, 1):
                    for start in range(len(run) - width + 1):
                        tokens = run[start : start + width]
                        lowered = [token.casefold() for token in tokens]
                        if width == 1 and (
                            lowered[0] in _GENERIC_CONCEPT_TERMS or len(lowered[0]) < 4
                        ):
                            continue
                        if (
                            lowered[0] in _GENERIC_CONCEPT_TERMS
                            or lowered[-1] in _GENERIC_CONCEPT_TERMS
                        ):
                            continue
                        if all(token in _GENERIC_CONCEPT_TERMS for token in lowered):
                            continue
                        key = _concept_key(tokens)
                        if not key:
                            continue
                        display = " ".join(tokens)
                        if len(display) > 120:
                            continue
                        previous = candidates.get(key)
                        if previous is None:
                            candidates[key] = (display, source_weight, 1)
                        else:
                            previous_display, previous_weight, count = previous
                            candidates[key] = (
                                previous_display,
                                max(previous_weight, source_weight),
                                count + 1,
                            )

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1][1],
            -item[1][2],
            -len(item[0].split()),
            item[0],
        ),
    )
    return [details[0] for _, details in ordered[:MAX_CONCEPT_CANDIDATES]]


def _meaningful_title_tokens(value: str) -> list[str]:
    return [token for token in normalize_title(value).split() if token not in _TITLE_STOPWORDS]


def _strong_title_coverage(query: str, title: str) -> bool:
    """Recognize conservative named/partial-title queries without boosting generic overlap."""
    query_tokens = _meaningful_title_tokens(query)
    title_tokens = set(_meaningful_title_tokens(title))
    if len(query_tokens) < 2 or not set(query_tokens).issubset(title_tokens):
        return False
    distinctive_tokens = {
        token for token in query_tokens if len(token) >= 5 and token not in _GENERIC_TITLE_TERMS
    }
    return bool(distinctive_tokens)


def information_completeness(
    paper: Paper,
    *,
    parsed_pdf_available: bool = False,
    structured_insights_available: bool = False,
) -> float:
    """Return the highest explicitly satisfied information-availability state."""
    if structured_insights_available:
        return INSIGHTS_COMPLETENESS
    if parsed_pdf_available:
        return PARSED_PDF_COMPLETENESS
    if paper.abstract and paper.abstract.strip():
        return ABSTRACT_COMPLETENESS
    return METADATA_COMPLETENESS


def _canonical_pair(first_id: str, second_id: str) -> tuple[str, str]:
    return (first_id, second_id) if first_id < second_id else (second_id, first_id)


def _validate_paper_ids(papers: Sequence[Paper]) -> None:
    paper_ids = [paper.id for paper in papers]
    if len(set(paper_ids)) != len(paper_ids):
        raise GraphSemanticsInputError("Graph papers must have unique paper IDs")


def select_sparse_edges(
    paper_ids: Sequence[str], similarity_matrix: np.ndarray
) -> list[GraphEdgeSeed]:
    """Select the deterministic undirected union of thresholded top-K neighbors."""
    count = len(paper_ids)
    if similarity_matrix.shape != (count, count):
        raise GraphSemanticsInputError("Paper similarity matrix has an unexpected shape")
    if len(set(paper_ids)) != count:
        raise GraphSemanticsInputError("Graph papers must have unique paper IDs")

    directed_selections: set[tuple[str, str]] = set()
    pair_similarity: dict[tuple[str, str], float] = {}
    for source_index, source_id in enumerate(paper_ids):
        neighbors = sorted(
            (index for index in range(count) if index != source_index),
            key=lambda index: (-float(similarity_matrix[source_index, index]), paper_ids[index]),
        )
        for target_index in neighbors[:K_NEIGHBORS]:
            similarity = _canonical_relevance(float(similarity_matrix[source_index, target_index]))
            if similarity < MIN_EDGE_SIMILARITY:
                continue
            target_id = paper_ids[target_index]
            directed_selections.add((source_id, target_id))
            pair_similarity[_canonical_pair(source_id, target_id)] = similarity

    edges: list[GraphEdgeSeed] = []
    for source_id, target_id in sorted({_canonical_pair(*pair) for pair in directed_selections}):
        source_selected = (source_id, target_id) in directed_selections
        target_selected = (target_id, source_id) in directed_selections
        if source_selected and target_selected:
            reason: SelectionReason = "both_top_k"
        elif source_selected:
            reason = "source_top_k"
        else:
            reason = "target_top_k"
        similarity = pair_similarity[(source_id, target_id)]
        edges.append(
            GraphEdgeSeed(
                source_paper_id=source_id,
                target_paper_id=target_id,
                paper_similarity=similarity,
                edge_weight=similarity,
                selection_reason=reason,
            )
        )
    return edges


def format_graph_seed_audit(seed: GraphSeed) -> str:
    """Return a stable developer-facing representation of frozen graph semantics."""
    lines = ["NODES:"]
    for node in seed.nodes:
        lines.append(
            "\t".join(
                [
                    node.paper_id,
                    node.title,
                    f"query_relevance={node.query_relevance:.6f}",
                    f"radius={node.node_radius:.6f}",
                    f"information_completeness={node.information_completeness:.2f}",
                    f"opacity={node.node_opacity:.2f}",
                ]
            )
        )
    lines.append("EDGES:")
    for edge in seed.edges:
        lines.append(
            "\t".join(
                [
                    edge.source_paper_id,
                    edge.target_paper_id,
                    f"paper_similarity={edge.paper_similarity:.6f}",
                    f"edge_weight={edge.edge_weight:.6f}",
                    f"selection_reason={edge.selection_reason}",
                ]
            )
        )
    return "\n".join(lines)


class GraphSemanticsService:
    """Compute graph-only query relevance without changing M1 ranking or paper objects."""

    def __init__(
        self,
        embedding_model_name: str,
        bi_encoder: BiEncoder | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self._bi_encoder = bi_encoder
        self._load_lock = threading.Lock()

    def _load_model(self) -> BiEncoder:
        if self._bi_encoder is not None:
            return self._bi_encoder
        with self._load_lock:
            if self._bi_encoder is not None:
                return self._bi_encoder
            try:
                from sentence_transformers import SentenceTransformer

                self._bi_encoder = SentenceTransformer(self.embedding_model_name)
            except Exception as exc:
                raise GraphSemanticsUnavailableError(
                    "The local embedding model could not be initialized for graph relevance"
                ) from exc
        return self._bi_encoder

    def build_node_seeds(
        self,
        original_query: str,
        papers: list[Paper],
        *,
        parsed_paper_ids: Set[str] | None = None,
        insight_paper_ids: Set[str] | None = None,
    ) -> list[GraphNodeSeed]:
        """Return seeds in input order, anchored only to the original user query."""
        query = original_query.strip()
        if not query:
            raise GraphSemanticsInputError("Original query must not be empty")
        if not papers:
            return []
        _validate_paper_ids(papers)
        parsed_ids = parsed_paper_ids or frozenset()
        insight_ids = insight_paper_ids or frozenset()

        relevance_by_index: dict[int, float] = {
            index: 1.0
            for index, paper in enumerate(papers)
            if is_exact_title_match(query, paper.title)
        }
        semantic_items = [
            (index, paper) for index, paper in enumerate(papers) if index not in relevance_by_index
        ]
        if semantic_items:
            encoder = self._load_model()
            titles = [paper.title for _, paper in semantic_items]
            documents = [_paper_representation(paper) for _, paper in semantic_items]
            inputs = [query, *titles, *documents]
            try:
                embeddings = np.asarray(
                    encoder.encode(
                        inputs,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        batch_size=16,
                    ),
                    dtype=float,
                )
                if embeddings.ndim != 2 or embeddings.shape[0] != len(inputs):
                    raise ValueError("unexpected embedding shape")
                item_count = len(semantic_items)
                title_similarities = embeddings[1 : item_count + 1] @ embeddings[0]
                document_similarities = embeddings[item_count + 1 :] @ embeddings[0]
            except Exception as exc:
                raise GraphSemanticsUnavailableError(
                    "The local embedding model failed while scoring graph relevance"
                ) from exc
            for (index, paper), title_similarity, document_similarity in zip(
                semantic_items,
                title_similarities.tolist(),
                document_similarities.tolist(),
                strict=True,
            ):
                lexical_floor = (
                    STRONG_TITLE_COVERAGE_FLOOR
                    if _strong_title_coverage(query, paper.title)
                    else 0.0
                )
                relevance_by_index[index] = max(
                    _canonical_relevance(title_similarity),
                    _canonical_relevance(document_similarity),
                    lexical_floor,
                )

        seeds: list[GraphNodeSeed] = []
        for index, paper in enumerate(papers):
            relevance = relevance_by_index[index]
            completeness = information_completeness(
                paper,
                parsed_pdf_available=paper.id in parsed_ids,
                structured_insights_available=paper.id in insight_ids,
            )
            seeds.append(
                GraphNodeSeed(
                    paper_id=paper.id,
                    title=paper.title,
                    query_relevance=relevance,
                    node_weight=relevance,
                    node_radius=relevance_to_radius(relevance),
                    information_completeness=completeness,
                    node_opacity=relevance_to_opacity(completeness),
                )
            )
        return seeds

    def related_concepts(
        self,
        original_query: str,
        papers: list[Paper],
        *,
        limit: int = MAX_RELATED_CONCEPTS,
    ) -> list[RelatedConcept]:
        """Rank result-set phrases against the original query in one encoder call."""
        query = original_query.strip()
        if not query:
            raise GraphSemanticsInputError("Original query must not be empty")
        if limit < 1:
            return []
        candidates = extract_concept_candidates(papers)
        if not candidates:
            return []
        encoder = self._load_model()
        inputs = [query, *candidates]
        try:
            embeddings = np.asarray(
                encoder.encode(
                    inputs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=64,
                ),
                dtype=float,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(inputs):
                raise ValueError("unexpected embedding shape")
            similarities = embeddings[1:] @ embeddings[0]
        except Exception as exc:
            raise GraphSemanticsUnavailableError(
                "The local embedding model failed while ranking related concepts"
            ) from exc

        ranked = sorted(
            (
                RelatedConcept(
                    text=text,
                    query_similarity=_canonical_relevance(similarity),
                )
                for text, similarity in zip(candidates, similarities.tolist(), strict=True)
            ),
            key=lambda concept: (-concept.query_similarity, concept.text.casefold()),
        )
        return ranked[: min(limit, MAX_RELATED_CONCEPTS)]

    def paper_similarity_matrix(self, papers: list[Paper]) -> np.ndarray:
        """Encode papers in one batch and return their symmetric cosine matrix."""
        if not papers:
            return np.empty((0, 0), dtype=float)
        _validate_paper_ids(papers)
        encoder = self._load_model()
        inputs = [_paper_representation(paper) for paper in papers]
        try:
            embeddings = np.asarray(
                encoder.encode(
                    inputs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=16,
                ),
                dtype=float,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(inputs):
                raise ValueError("unexpected embedding shape")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms == 0) or not np.all(np.isfinite(norms)):
                raise ValueError("invalid embedding norm")
            normalized = embeddings / norms
            similarities = normalized @ normalized.T
            if not np.all(np.isfinite(similarities)):
                raise ValueError("non-finite paper similarity")
            return np.clip(similarities, 0.0, 1.0)
        except Exception as exc:
            raise GraphSemanticsUnavailableError(
                "The local embedding model failed while scoring paper similarity"
            ) from exc

    def build_edge_seeds(self, papers: list[Paper]) -> list[GraphEdgeSeed]:
        """Encode all papers once and return sparse query-independent semantic edges."""
        similarities = self.paper_similarity_matrix(papers)
        return select_sparse_edges([paper.id for paper in papers], similarities)

    def build_graph_seed(
        self,
        original_query: str,
        papers: list[Paper],
        *,
        parsed_paper_ids: Set[str] | None = None,
        insight_paper_ids: Set[str] | None = None,
    ) -> GraphSeed:
        """Build backend graph data without layout, API, or frontend concerns."""
        nodes = self.build_node_seeds(
            original_query,
            papers,
            parsed_paper_ids=parsed_paper_ids,
            insight_paper_ids=insight_paper_ids,
        )
        edges = self.build_edge_seeds(papers)
        related_concepts = self.related_concepts(original_query, papers)
        return GraphSeed(
            embedding_model=self.embedding_model_name,
            nodes=nodes,
            edges=edges,
            related_concepts=related_concepts,
        )


__all__ = [
    "ABSTRACT_COMPLETENESS",
    "GRAPH_SEMANTICS_VERSION",
    "INSIGHTS_COMPLETENESS",
    "K_NEIGHBORS",
    "MAX_CONCEPT_CANDIDATES",
    "MAX_RELATED_CONCEPTS",
    "MAX_RADIUS",
    "METADATA_COMPLETENESS",
    "MIN_EDGE_SIMILARITY",
    "MIN_NODE_OPACITY",
    "MIN_RADIUS",
    "PARSED_PDF_COMPLETENESS",
    "STRONG_TITLE_COVERAGE_FLOOR",
    "GraphEdgeSeed",
    "GraphNodeSeed",
    "GraphSeed",
    "GraphSemanticsInputError",
    "GraphSemanticsService",
    "GraphSemanticsUnavailableError",
    "RelatedConcept",
    "extract_concept_candidates",
    "format_graph_seed_audit",
    "information_completeness",
    "relevance_to_radius",
    "relevance_to_opacity",
    "select_sparse_edges",
]
