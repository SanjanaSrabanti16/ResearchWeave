# ruff: noqa: E402 -- validation script adds the backend package to sys.path.
from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app.services.ranking_service as ranking_module
from app.core.config import Settings
from app.db.database import create_session_factory, initialize_database
from app.models.api import SearchRequest
from app.providers import ArxivProvider, OpenAlexProvider, SemanticScholarProvider
from app.services.cache_service import CacheService
from app.services.deduplication_service import DeduplicationService
from app.services.query_variants import QueryVariantService
from app.services.ranking_service import RankingService
from app.services.search_service import SearchService

QUERY = "Agentic Design Patterns: A System-Theoretic Framework"
START_YEAR = 2025
END_YEAR = 2026
LIMIT = 20

CURRENT_CALL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "search_latency_call", default="unknown"
)


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.normalization_seconds: dict[str, float] = defaultdict(float)
        self.normalization_count: dict[str, int] = defaultdict(int)
        self.http_requests: list[dict[str, Any]] = []
        self.request_indexes: dict[int, int] = {}
        self.variant_calls: list[dict[str, Any]] = []
        self.cache_seconds = 0.0
        self.cache_reads = 0
        self.cache_writes = 0
        self.query_expansion_seconds = 0.0
        self.variants: list[str] = []
        self.dedupe_seconds = 0.0
        self.ranking_seconds = 0.0
        self.bi_encoder_query_seconds = 0.0
        self.bi_encoder_papers_seconds = 0.0
        self.cross_encoder_seconds = 0.0
        self.rrf_seconds = 0.0

    def add_normalization(self, provider: str, elapsed: float) -> None:
        with self.lock:
            self.normalization_seconds[provider] += elapsed
            self.normalization_count[provider] += 1


METRICS = Metrics()


class TimedOpenAlexProvider(OpenAlexProvider):
    @classmethod
    def normalize(cls, item: dict[str, Any]):
        started = time.perf_counter()
        try:
            return super().normalize(item)
        finally:
            METRICS.add_normalization(cls.name, time.perf_counter() - started)


class TimedSemanticScholarProvider(SemanticScholarProvider):
    @classmethod
    def normalize(cls, item: dict[str, Any]):
        started = time.perf_counter()
        try:
            return super().normalize(item)
        finally:
            METRICS.add_normalization(cls.name, time.perf_counter() - started)


class TimedArxivProvider(ArxivProvider):
    @classmethod
    def normalize(cls, result):
        started = time.perf_counter()
        try:
            return super().normalize(result)
        finally:
            METRICS.add_normalization(cls.name, time.perf_counter() - started)


class TrackingRequestsSession(requests.Session):
    def request(self, method: str, url: str, **kwargs: Any):
        started = time.perf_counter()
        record = {
            "provider": "arxiv",
            "call": CURRENT_CALL.get(),
            "method": method,
            "host": httpx.URL(url).host,
            "started": started,
            "ended": None,
            "status": None,
            "error": None,
        }
        with METRICS.lock:
            METRICS.http_requests.append(record)
        try:
            response = super().request(method, url, **kwargs)
            record["status"] = response.status_code
            return response
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["ended"] = time.perf_counter()


class TimedProvider:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.name = provider.name

    async def search(self, query: str, limit: int, *years: Any):
        call = f"{self.name}:{query}"
        token = CURRENT_CALL.set(call)
        started = time.perf_counter()
        try:
            papers = await self.provider.search(query, limit, *years)
            outcome = "ok"
            return papers
        except Exception as exc:
            outcome = f"error:{type(exc).__name__}"
            raise
        finally:
            ended = time.perf_counter()
            METRICS.variant_calls.append(
                {
                    "provider": self.name,
                    "query": query,
                    "inner_started": started,
                    "inner_ended": ended,
                    "provider_call_seconds": ended - started,
                    "outcome": outcome,
                }
            )
            CURRENT_CALL.reset(token)


class TimedCache(CacheService):
    def get(self, *args: Any):
        started = time.perf_counter()
        try:
            return super().get(*args)
        finally:
            METRICS.cache_seconds += time.perf_counter() - started
            METRICS.cache_reads += 1

    def set(self, *args: Any):
        started = time.perf_counter()
        try:
            return super().set(*args)
        finally:
            METRICS.cache_seconds += time.perf_counter() - started
            METRICS.cache_writes += 1


class TimedVariants(QueryVariantService):
    def generate(self, query: str) -> list[str]:
        started = time.perf_counter()
        result = super().generate(query)
        METRICS.query_expansion_seconds = time.perf_counter() - started
        METRICS.variants = result
        return result


class TimedDeduplicator(DeduplicationService):
    def deduplicate(self, papers):
        started = time.perf_counter()
        try:
            return super().deduplicate(papers)
        finally:
            METRICS.dedupe_seconds = time.perf_counter() - started


class TimedEncoder:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def encode(self, sentences: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return self.delegate.encode(sentences, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            if isinstance(sentences, str):
                METRICS.bi_encoder_query_seconds += elapsed
            else:
                METRICS.bi_encoder_papers_seconds += elapsed


class TimedReranker:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def predict(self, sentences: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return self.delegate.predict(sentences, **kwargs)
        finally:
            METRICS.cross_encoder_seconds += time.perf_counter() - started


class TimedRankingService(RankingService):
    def rank(self, query, papers, limit):
        started = time.perf_counter()
        try:
            return super().rank(query, papers, limit)
        finally:
            METRICS.ranking_seconds = time.perf_counter() - started


class TimedSearchService(SearchService):
    async def _search_provider(self, provider, request, provider_limit):
        outer_started = time.perf_counter()
        call = f"{provider.name}:{request.query}"
        token = CURRENT_CALL.set(call)
        try:
            return await super()._search_provider(provider, request, provider_limit)
        finally:
            outer_ended = time.perf_counter()
            for record in reversed(METRICS.variant_calls):
                if record["provider"] == provider.name and record["query"] == request.query:
                    record["outer_started"] = outer_started
                    record["outer_ended"] = outer_ended
                    record["total_variant_seconds"] = outer_ended - outer_started
                    break
            CURRENT_CALL.reset(token)


def retry_wait_seconds(records: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["call"]].append(record)
    total = 0.0
    for attempts in grouped.values():
        attempts.sort(key=lambda item: item["started"])
        for previous, current in zip(attempts, attempts[1:], strict=False):
            if previous["ended"] is not None:
                total += max(0.0, current["started"] - previous["ended"])
    return total


async def main() -> None:
    settings = Settings()
    query_prep_started = time.perf_counter()
    request = SearchRequest(
        query=QUERY,
        start_year=START_YEAR,
        end_year=END_YEAR,
        limit=LIMIT,
    )
    query_preparation_seconds = time.perf_counter() - query_prep_started

    request_map: dict[int, dict[str, Any]] = {}

    async def on_request(http_request: httpx.Request) -> None:
        record = {
            "provider": CURRENT_CALL.get().split(":", 1)[0],
            "call": CURRENT_CALL.get(),
            "method": http_request.method,
            "host": http_request.url.host,
            "started": time.perf_counter(),
            "ended": None,
            "status": None,
            "error": None,
        }
        request_map[id(http_request)] = record
        METRICS.http_requests.append(record)

    async def on_response(response: httpx.Response) -> None:
        record = request_map[id(response.request)]
        record["status"] = response.status_code
        record["ended"] = time.perf_counter()

    timeout = httpx.Timeout(settings.provider_timeout_seconds)
    with tempfile.TemporaryDirectory(
        prefix="researchweave-search-diagnostic-", ignore_cleanup_errors=True
    ) as directory:
        database_url = f"sqlite:///{Path(directory) / 'cache.sqlite3'}"
        session_factory = create_session_factory(database_url)
        initialize_database(session_factory)
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "research-landscape-explorer/0.1 (open-source research tool)"},
            follow_redirects=True,
            event_hooks={"request": [on_request], "response": [on_response]},
        ) as client:
            arxiv_provider = TimedArxivProvider(client, retries=settings.provider_retries)
            tracking_session = TrackingRequestsSession()
            arxiv_provider.arxiv_client._session = tracking_session
            providers = [
                TimedProvider(
                    TimedOpenAlexProvider(
                        client,
                        retries=settings.provider_retries,
                        api_key=settings.openalex_api_key,
                    )
                ),
                TimedProvider(
                    TimedSemanticScholarProvider(
                        client,
                        retries=settings.provider_retries,
                        api_key=settings.semantic_scholar_api_key,
                    )
                ),
                TimedProvider(arxiv_provider),
            ]
            ranker = TimedRankingService(
                settings.embedding_model,
                settings.reranker_model,
                settings.rerank_shortlist_size,
                fusion_mode=settings.ranking_fusion_mode,
            )
            model_load_started = time.perf_counter()
            ranker._load_models()
            model_load_seconds = time.perf_counter() - model_load_started
            ranker._bi_encoder = TimedEncoder(ranker._bi_encoder)
            ranker._reranker = TimedReranker(ranker._reranker)

            original_rrf = ranking_module.reciprocal_rank_fusion

            def timed_rrf(*args: Any, **kwargs: Any):
                started = time.perf_counter()
                try:
                    return original_rrf(*args, **kwargs)
                finally:
                    METRICS.rrf_seconds += time.perf_counter() - started

            ranking_module.reciprocal_rank_fusion = timed_rrf
            service = TimedSearchService(
                providers=providers,
                cache=TimedCache(session_factory, settings.cache_ttl_hours),
                deduplicator=TimedDeduplicator(),
                ranker=ranker,
                candidate_target=settings.candidate_target,
                query_variants=TimedVariants(),
            )
            search_started = time.perf_counter()
            try:
                response = await service.search(request)
            finally:
                search_seconds = time.perf_counter() - search_started
                ranking_module.reciprocal_rank_fusion = original_rrf
                tracking_session.close()

    outer_starts = [item["outer_started"] for item in METRICS.variant_calls]
    outer_ends = [item["outer_ended"] for item in METRICS.variant_calls]
    provider_wall_seconds = max(outer_ends) - min(outer_starts)
    response_assembly_seconds = max(
        0.0,
        search_seconds
        - METRICS.query_expansion_seconds
        - provider_wall_seconds
        - METRICS.dedupe_seconds
        - METRICS.ranking_seconds,
    )
    provider_totals: dict[str, float] = {}
    for provider in ("openalex", "semantic_scholar", "arxiv"):
        calls = [item for item in METRICS.variant_calls if item["provider"] == provider]
        provider_totals[provider] = sum(item["total_variant_seconds"] for item in calls)

    request_summary: dict[str, Any] = {}
    for provider in ("openalex", "semantic_scholar", "arxiv"):
        records = [item for item in METRICS.http_requests if item["provider"] == provider]
        statuses: dict[str, int] = defaultdict(int)
        for record in records:
            key = str(
                record["status"]
                if record["status"] is not None
                else record["error"] or "no_response"
            )
            statuses[key] += 1
        provider_calls = [item for item in METRICS.variant_calls if item["provider"] == provider]
        request_summary[provider] = {
            "variant_calls": len(provider_calls),
            "http_requests": len(records),
            "retries": max(0, len(records) - len(provider_calls)),
            "statuses": dict(statuses),
            "timeouts_or_no_response": sum(1 for record in records if record["status"] is None),
            "retry_or_internal_rate_wait_seconds": round(retry_wait_seconds(records), 6),
            "failed_variant_calls": sum(1 for call in provider_calls if call["outcome"] != "ok"),
        }

    output = {
        "query": QUERY,
        "years": [START_YEAR, END_YEAR],
        "limit": LIMIT,
        "variants": METRICS.variants,
        "settings": {
            "provider_timeout_seconds": settings.provider_timeout_seconds,
            "provider_retries": settings.provider_retries,
            "arxiv_num_retries": arxiv_provider.arxiv_client.num_retries,
            "arxiv_delay_seconds": arxiv_provider.arxiv_client.delay_seconds,
            "ranking_profile": settings.ranking_profile,
            "fusion_mode": settings.ranking_fusion_mode,
        },
        "model_load_preflight_seconds_excluded_from_search": round(model_load_seconds, 6),
        "timings_seconds": {
            "query_preparation": round(query_preparation_seconds, 6),
            "query_expansion": round(METRICS.query_expansion_seconds, 6),
            "provider_wall": round(provider_wall_seconds, 6),
            "openalex_total": round(provider_totals["openalex"], 6),
            "semantic_scholar_total": round(provider_totals["semantic_scholar"], 6),
            "arxiv_total": round(provider_totals["arxiv"], 6),
            "provider_retry_or_internal_rate_wait": round(
                retry_wait_seconds(METRICS.http_requests), 6
            ),
            "cache_lookup_and_write_cpu": round(METRICS.cache_seconds, 6),
            "normalization_cpu": round(sum(METRICS.normalization_seconds.values()), 6),
            "deduplication_canonical_merge": round(METRICS.dedupe_seconds, 6),
            "ranking_total": round(METRICS.ranking_seconds, 6),
            "bi_encoder": round(
                METRICS.bi_encoder_query_seconds + METRICS.bi_encoder_papers_seconds, 6
            ),
            "cross_encoder": round(METRICS.cross_encoder_seconds, 6),
            "rrf": round(METRICS.rrf_seconds, 6),
            "response_assembly_and_filtering": round(response_assembly_seconds, 6),
            "search_total": round(search_seconds, 6),
            "total_including_request_parsing": round(query_preparation_seconds + search_seconds, 6),
        },
        "normalization": {
            provider: {
                "count": METRICS.normalization_count[provider],
                "seconds": round(METRICS.normalization_seconds[provider], 6),
            }
            for provider in ("openalex", "semantic_scholar", "arxiv")
        },
        "provider_requests": request_summary,
        "variant_calls": [
            {
                "provider": item["provider"],
                "query": item["query"],
                "seconds": round(item["total_variant_seconds"], 6),
                "provider_call_seconds": round(item["provider_call_seconds"], 6),
                "outcome": item["outcome"],
            }
            for item in METRICS.variant_calls
        ],
        "result": {
            "candidate_count": response.candidate_count,
            "deduplicated_count": response.deduplicated_count,
            "ranked_count": response.ranked_count,
            "provider_status": response.provider_status,
            "warnings": response.warnings,
        },
        "cache": {"reads": METRICS.cache_reads, "writes": METRICS.cache_writes},
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
