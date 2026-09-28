from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.api import CachedPaperAnalysisResponse, GraphRequest
from app.models.document import ParsedPaper, SourcePDF
from app.models.insights import InsightFields, InsightsResponse
from app.models.paper import Paper
from app.services.graph_semantics import (
    GraphEdgeSeed,
    GraphNodeSeed,
    GraphSeed,
    GraphSemanticsUnavailableError,
)
from app.services.graph_service import GraphService, GraphStateResolver
from app.services.insight_cache import InsightCache
from app.services.insight_service import document_fingerprint
from app.services.paper_understanding_service import PIPELINE_VERSION
from app.services.parsed_document_cache import ParsedDocumentCache


def _seed() -> GraphSeed:
    return GraphSeed(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        nodes=[
            GraphNodeSeed(
                paper_id="paper-a",
                title="Paper A",
                query_relevance=0.9,
                node_weight=0.9,
                node_radius=26.412118,
                information_completeness=1.0,
                node_opacity=1.0,
            ),
            GraphNodeSeed(
                paper_id="paper-b",
                title="Paper B",
                query_relevance=0.5,
                node_weight=0.5,
                node_radius=20.59126,
                information_completeness=0.78,
                node_opacity=0.78,
            ),
            GraphNodeSeed(
                paper_id="paper-isolated",
                title="Isolated",
                query_relevance=0.2,
                node_weight=0.2,
                node_radius=14.96663,
                information_completeness=0.4,
                node_opacity=0.4,
            ),
        ],
        edges=[
            GraphEdgeSeed(
                source_paper_id="paper-a",
                target_paper_id="paper-b",
                paper_similarity=0.82,
                edge_weight=0.82,
                selection_reason="both_top_k",
            )
        ],
    )


class RecordingGraphService:
    def __init__(self) -> None:
        self.requests: list[GraphRequest] = []

    async def build(self, request: GraphRequest) -> GraphSeed:
        self.requests.append(request)
        return _seed()

    async def cached_analysis(self, paper_id: str) -> CachedPaperAnalysisResponse:
        return CachedPaperAnalysisResponse(paper_id=paper_id)


class FailingGraphService(RecordingGraphService):
    async def build(self, request: GraphRequest) -> GraphSeed:
        raise GraphSemanticsUnavailableError("Graph embeddings unavailable")


def _papers() -> list[dict[str, object]]:
    return [
        Paper(id="paper-a", title="Paper A", semantic_score=0.9).model_dump(mode="json"),
        Paper(id="paper-b", title="Paper B", semantic_score=0.5).model_dump(mode="json"),
        Paper(id="paper-isolated", title="Isolated", semantic_score=0.2).model_dump(mode="json"),
    ]


def test_graph_endpoint_preserves_frozen_semantics_and_all_selected_nodes(tmp_path) -> None:
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}"))
    graph_service = RecordingGraphService()
    original_papers = _papers()
    with TestClient(app) as client:
        app.state.graph_service = graph_service
        response = client.post(
            "/api/graph",
            json={"query": "agent systems", "papers": original_papers},
        )

    assert response.status_code == 200
    payload = response.json()
    assert [node["paper_id"] for node in payload["nodes"]] == [
        "paper-a",
        "paper-b",
        "paper-isolated",
    ]
    assert payload["nodes"][0]["query_relevance"] == 0.9
    assert payload["nodes"][0]["node_radius"] == 26.412118
    assert payload["nodes"][0]["information_completeness"] == 1.0
    assert payload["nodes"][0]["node_opacity"] == 1.0
    assert payload["edges"] == [
        {
            "source_paper_id": "paper-a",
            "target_paper_id": "paper-b",
            "paper_similarity": 0.82,
            "edge_weight": 0.82,
            "selection_reason": "both_top_k",
        }
    ]
    assert "embedding" not in str(payload["nodes"]).casefold()
    assert "embedding" not in str(payload["edges"]).casefold()
    assert original_papers == _papers()
    assert [paper.id for paper in graph_service.requests[0].papers] == [
        "paper-a",
        "paper-b",
        "paper-isolated",
    ]


def test_graph_failure_is_isolated_as_safe_503(tmp_path) -> None:
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}"))
    with TestClient(app) as client:
        app.state.graph_service = FailingGraphService()
        response = client.post(
            "/api/graph",
            json={"query": "agent systems", "papers": _papers()},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Graph embeddings unavailable"}


def test_cached_analysis_is_never_served_from_browser_or_proxy_cache(tmp_path) -> None:
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}"))
    with TestClient(app) as client:
        app.state.graph_service = RecordingGraphService()
        response = client.get("/api/papers/paper-a/cached-analysis")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_graph_state_resolver_uses_real_parsed_and_current_insight_cache(tmp_path) -> None:
    parsed_cache = ParsedDocumentCache(tmp_path / "parsed")
    insight_cache = InsightCache(tmp_path / "insights")
    document = ParsedPaper(
        paper_id="paper-a",
        title="Paper A",
        parser="grobid",
        parser_version="0.9.1",
        source_pdf=SourcePDF(
            acquisition_method="upload",
            sha256="a" * 64,
            size_bytes=100,
        ),
    )
    parsed_cache.put("document-key", document)
    fingerprint = document_fingerprint(document)
    insight_cache.put(
        insight_cache.key(fingerprint, "test-model", PIPELINE_VERSION, "ollama"),
        InsightFields.empty(),
    )
    registry = SimpleNamespace(
        statuses=lambda: [
            {
                "provider_id": "ollama",
                "model": "test-model",
                "configured": True,
                "cloud": False,
            }
        ]
    )
    resolver = GraphStateResolver(tmp_path / "parsed", insight_cache, registry)

    state = resolver.resolve({"paper-a", "paper-b"})

    assert state["paper-a"].document == document
    assert state["paper-a"].insights is not None
    assert state["paper-a"].insights.cached is True
    assert state["paper-a"].insight_provider == "ollama"
    assert "paper-b" not in state


def test_graph_state_resolver_prefers_authoritative_current_document_and_insight(tmp_path) -> None:
    parsed_cache = ParsedDocumentCache(tmp_path / "parsed")
    insight_cache = InsightCache(tmp_path / "insights")
    current_document = ParsedPaper(
        paper_id="paper-a",
        title="Current parsed title",
        parser="grobid",
        parser_version="0.9.1",
        source_pdf=SourcePDF(acquisition_method="upload", sha256="a" * 64, size_bytes=100),
    )
    legacy_document = current_document.model_copy(update={"title": "Stale legacy title"})
    parsed_cache.put_paper_state(current_document)
    parsed_cache.put("later-content-key", legacy_document)
    fingerprint = document_fingerprint(current_document)
    ollama_insights = InsightFields.empty()
    evl_insights = InsightFields.empty()
    insight_cache.put(
        insight_cache.key(fingerprint, "qwen3:1.7b", PIPELINE_VERSION, "ollama"),
        ollama_insights,
    )
    insight_cache.put(
        insight_cache.key(fingerprint, "gemma4", PIPELINE_VERSION, "evl_gemma"),
        evl_insights,
    )
    insight_cache.put_current(
        paper_id="paper-a",
        document_fingerprint=fingerprint,
        provider_id="evl_gemma",
        model="gemma4",
        extraction_version=PIPELINE_VERSION,
        insights=evl_insights,
    )
    registry = SimpleNamespace(
        statuses=lambda: [
            {"provider_id": "ollama", "model": "qwen3:1.7b", "configured": True},
            {"provider_id": "evl_gemma", "model": "gemma4", "configured": True},
        ]
    )

    state = GraphStateResolver(tmp_path / "parsed", insight_cache, registry).resolve({"paper-a"})

    assert state["paper-a"].document.title == "Current parsed title"
    assert state["paper-a"].insight_provider == "evl_gemma"
    assert state["paper-a"].insights is not None
    assert state["paper-a"].insights.model == "gemma4"


def test_current_insight_state_rejects_stale_fingerprint_or_pipeline(tmp_path) -> None:
    cache = InsightCache(tmp_path)
    cache.put_current(
        paper_id="paper-a",
        document_fingerprint="a" * 64,
        provider_id="ollama",
        model="qwen3:1.7b",
        extraction_version=PIPELINE_VERSION,
        insights=InsightFields.empty(),
    )

    assert cache.get_current("paper-a", "b" * 64, PIPELINE_VERSION) is None
    assert cache.get_current("paper-a", "a" * 64, "future-version") is None


def test_graph_service_maps_parsed_and_insight_ids_without_provider_search() -> None:
    seed = _seed()
    semantics = SimpleNamespace()
    calls: list[dict[str, object]] = []

    def build_graph_seed(query, papers, **kwargs):
        calls.append({"query": query, "papers": papers, **kwargs})
        return seed

    semantics.build_graph_seed = build_graph_seed
    resolver = SimpleNamespace(
        resolve=lambda _: {
            "paper-a": CachedPaperAnalysisResponse(
                paper_id="paper-a",
                insights=InsightsResponse(
                    paper_id="paper-a",
                    document_fingerprint="a" * 64,
                    model="test-model",
                    extraction_version=PIPELINE_VERSION,
                    cached=True,
                    insights=InsightFields.empty(),
                ),
            ),
            "paper-b": CachedPaperAnalysisResponse(paper_id="paper-b"),
        }
    )
    service = GraphService(semantics, resolver)
    request = GraphRequest(
        query="agent systems",
        papers=[Paper(id="paper-a", title="A"), Paper(id="paper-b", title="B")],
    )

    result = asyncio.run(service.build(request))

    assert result == seed
    assert calls[0]["parsed_paper_ids"] == {"paper-a", "paper-b"}
    assert calls[0]["insight_paper_ids"] == {"paper-a"}
    assert [paper.id for paper in request.papers] == ["paper-a", "paper-b"]
