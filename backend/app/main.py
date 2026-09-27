from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.db.database import create_session_factory, initialize_database
from app.llm import EVLGemmaProvider, GeminiProvider, LLMProviderRegistry, OllamaProvider
from app.parsers import GrobidParser
from app.providers import ArxivProvider, OpenAlexProvider, SemanticScholarProvider
from app.providers.unpaywall import UnpaywallProvider
from app.services.cache_service import CacheService
from app.services.deduplication_service import DeduplicationService
from app.services.insight_cache import InsightCache
from app.services.paper_understanding_service import PaperUnderstandingService
from app.services.parsed_document_cache import ParsedDocumentCache
from app.services.pdf_service import PDFAcquisitionService, PDFProcessingService
from app.services.pdf_validation import SecurePDFDownloader
from app.services.ranking_service import RankingService
from app.services.search_service import SearchService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        session_factory = create_session_factory(settings.database_url)
        initialize_database(session_factory)
        timeout = httpx.Timeout(settings.provider_timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "research-landscape-explorer/0.1 (open-source research tool)"},
            follow_redirects=True,
        ) as client:
            arxiv_provider = ArxivProvider(client, retries=settings.provider_retries)
            providers = [
                OpenAlexProvider(
                    client,
                    retries=settings.provider_retries,
                    api_key=settings.openalex_api_key,
                ),
                SemanticScholarProvider(
                    client,
                    retries=settings.provider_retries,
                    api_key=settings.semantic_scholar_api_key,
                ),
                arxiv_provider,
            ]
            app.state.search_service = SearchService(
                providers=providers,
                cache=CacheService(session_factory, settings.cache_ttl_hours),
                deduplicator=DeduplicationService(),
                ranker=RankingService(
                    settings.embedding_model,
                    settings.reranker_model,
                    settings.rerank_shortlist_size,
                    fusion_mode=settings.ranking_fusion_mode,
                ),
                candidate_target=settings.candidate_target,
            )
            max_pdf_bytes = settings.pdf_max_size_mb * 1024 * 1024
            processor = PDFProcessingService(
                parser=GrobidParser(
                    client,
                    base_url=settings.grobid_url,
                    version=settings.grobid_parser_version,
                    timeout_seconds=settings.grobid_timeout_seconds,
                ),
                cache=ParsedDocumentCache(settings.parsed_document_cache_dir),
                max_bytes=max_pdf_bytes,
            )
            app.state.pdf_service = PDFAcquisitionService(
                downloader=SecurePDFDownloader(
                    client,
                    max_bytes=max_pdf_bytes,
                    timeout_seconds=settings.pdf_download_timeout_seconds,
                    allowed_hosts=settings.pdf_allowed_host_list,
                ),
                unpaywall=UnpaywallProvider(client, settings.unpaywall_email),
                processor=processor,
                arxiv_provider=arxiv_provider,
            )
            async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as ollama_client:
                registry = LLMProviderRegistry(
                    providers=(
                        OllamaProvider(
                            client=ollama_client,
                            base_url=settings.ollama_base_url,
                            model_id=settings.ollama_model,
                        ),
                        GeminiProvider(
                            api_key=(
                                settings.gemini_api_key.get_secret_value()
                                if settings.gemini_api_key is not None
                                else None
                            ),
                            model_id=settings.gemini_model,
                            timeout_seconds=settings.gemini_timeout_seconds,
                        ),
                        EVLGemmaProvider(
                            api_key=(
                                settings.evl_gemma_api_key.get_secret_value()
                                if settings.evl_gemma_api_key is not None
                                else None
                            ),
                            base_url=settings.evl_gemma_base_url,
                            model_id=settings.evl_gemma_model,
                            timeout_seconds=settings.evl_gemma_timeout_seconds,
                            diagnostic_dir=settings.llm_diagnostic_dir,
                        ),
                    ),
                    default_provider=settings.llm_provider,
                )
                app.state.llm_registry = registry
                app.state.insight_service = PaperUnderstandingService(
                    registry=registry,
                    cache=InsightCache(settings.insight_cache_dir),
                    batch_chars=settings.ollama_batch_chars,
                )
                try:
                    yield
                finally:
                    await registry.aclose()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Free scholarly retrieval, local relevance ranking, and open-access PDF parsing."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
