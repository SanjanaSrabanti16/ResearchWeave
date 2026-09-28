from __future__ import annotations

import asyncio
from pathlib import Path

from app.llm import LLMProviderRegistry
from app.models.api import CachedPaperAnalysisResponse, GraphRequest
from app.models.document import ParsedPaper
from app.models.insights import InsightsResponse
from app.services.graph_semantics import GraphSeed, GraphSemanticsService
from app.services.insight_cache import InsightCache
from app.services.insight_service import document_fingerprint, validate_insights
from app.services.paper_understanding_service import PIPELINE_VERSION
from app.services.parsed_document_cache import ParsedDocumentCache


class GraphStateResolver:
    """Read existing parsed-document and insight caches by canonical paper ID."""

    def __init__(
        self,
        parsed_document_directory: str | Path,
        insight_cache: InsightCache,
        registry: LLMProviderRegistry,
    ) -> None:
        self.parsed_document_directory = Path(parsed_document_directory)
        self.parsed_document_cache = ParsedDocumentCache(parsed_document_directory)
        self.insight_cache = insight_cache
        self.registry = registry

    def _documents(self, paper_ids: set[str]) -> dict[str, ParsedPaper]:
        return self.parsed_document_cache.resolve_paper_states(paper_ids)

    def resolve(self, paper_ids: set[str]) -> dict[str, CachedPaperAnalysisResponse]:
        documents = self._documents(paper_ids)
        states: dict[str, CachedPaperAnalysisResponse] = {}
        provider_statuses = self.registry.statuses()
        configured_providers = {
            (str(provider["provider_id"]), str(provider["model"]))
            for provider in provider_statuses
            if bool(provider["configured"])
        }
        for paper_id, document in documents.items():
            fingerprint = document_fingerprint(document)
            cached_insights = None
            cached_provider = None
            cached_model = None
            current = self.insight_cache.get_current(paper_id, fingerprint, PIPELINE_VERSION)
            if (
                current is not None
                and (current.provider_id, current.model) in configured_providers
                and validate_insights(current.insights, document) == current.insights
            ):
                cached_insights = current.insights
                cached_provider = current.provider_id
                cached_model = current.model
            else:
                for provider in provider_statuses:
                    if not bool(provider["configured"]):
                        continue
                    provider_id = str(provider["provider_id"])
                    model = str(provider["model"])
                    key = self.insight_cache.key(
                        fingerprint,
                        model,
                        PIPELINE_VERSION,
                        provider_id,
                    )
                    insights = self.insight_cache.get(key)
                    if insights is not None and validate_insights(insights, document) == insights:
                        cached_insights = insights
                        cached_provider = provider_id
                        cached_model = model
                        break
            insight_response = None
            if cached_insights is not None and cached_model is not None:
                insight_response = InsightsResponse(
                    paper_id=paper_id,
                    document_fingerprint=fingerprint,
                    model=cached_model,
                    extraction_version=PIPELINE_VERSION,
                    cached=True,
                    insights=cached_insights,
                )
            states[paper_id] = CachedPaperAnalysisResponse(
                paper_id=paper_id,
                document=document,
                insights=insight_response,
                insight_provider=cached_provider,
            )
        return states


class GraphService:
    """Build graph data from already selected papers without invoking retrieval providers."""

    def __init__(
        self,
        semantics: GraphSemanticsService,
        state_resolver: GraphStateResolver,
    ) -> None:
        self.semantics = semantics
        self.state_resolver = state_resolver

    async def build(self, request: GraphRequest) -> GraphSeed:
        paper_ids = {paper.id for paper in request.papers}
        states = await asyncio.to_thread(self.state_resolver.resolve, paper_ids)
        parsed_ids = set(states)
        insight_ids = {paper_id for paper_id, state in states.items() if state.insights is not None}
        return await asyncio.to_thread(
            self.semantics.build_graph_seed,
            request.query,
            request.papers,
            parsed_paper_ids=parsed_ids,
            insight_paper_ids=insight_ids,
        )

    async def cached_analysis(self, paper_id: str) -> CachedPaperAnalysisResponse:
        states = await asyncio.to_thread(self.state_resolver.resolve, {paper_id})
        return states.get(paper_id, CachedPaperAnalysisResponse(paper_id=paper_id))


__all__ = ["GraphService", "GraphStateResolver"]
