# ruff: noqa: E402 -- validation script adds the backend package to sys.path.
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from diagnose_search_latency import (
    END_YEAR,
    LIMIT,
    METRICS,
    QUERY,
    START_YEAR,
    TimedCache,
    TimedDeduplicator,
    TimedEncoder,
    TimedRankingService,
    TimedReranker,
    TimedVariants,
)

import app.services.ranking_service as ranking_module
from app.core.config import Settings
from app.db.database import create_session_factory
from app.models.api import SearchRequest
from app.providers.base import ProviderError
from app.services.search_service import SearchService

CACHE_PATH = Path(
    os.environ.get("RESEARCHWEAVE_REPLAY_CACHE", PROJECT_ROOT / "review_live_cache.sqlite3")
)


class NoNetworkProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def search(self, *_: object):
        self.calls += 1
        raise ProviderError(f"{self.name} cache miss during offline replay")


async def main() -> None:
    settings = Settings()
    providers = [
        NoNetworkProvider("openalex"),
        NoNetworkProvider("semantic_scholar"),
        NoNetworkProvider("arxiv"),
    ]
    ranker = TimedRankingService(
        settings.embedding_model,
        settings.reranker_model,
        settings.rerank_shortlist_size,
        fusion_mode=settings.ranking_fusion_mode,
    )
    load_started = time.perf_counter()
    ranker._load_models()
    model_load_seconds = time.perf_counter() - load_started
    ranker._bi_encoder = TimedEncoder(ranker._bi_encoder)
    ranker._reranker = TimedReranker(ranker._reranker)

    original_rrf = ranking_module.reciprocal_rank_fusion

    def timed_rrf(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_rrf(*args, **kwargs)
        finally:
            METRICS.rrf_seconds += time.perf_counter() - started

    ranking_module.reciprocal_rank_fusion = timed_rrf
    service = SearchService(
        providers=providers,
        cache=TimedCache(create_session_factory(f"sqlite:///{CACHE_PATH}"), 24),
        deduplicator=TimedDeduplicator(),
        ranker=ranker,
        candidate_target=settings.candidate_target,
        query_variants=TimedVariants(),
    )
    started = time.perf_counter()
    try:
        response = await service.search(
            SearchRequest(
                query=QUERY,
                start_year=START_YEAR,
                end_year=END_YEAR,
                limit=LIMIT,
            )
        )
    finally:
        total = time.perf_counter() - started
        ranking_module.reciprocal_rank_fusion = original_rrf

    output = {
        "network_requests": 0,
        "model_load_seconds_excluded": round(model_load_seconds, 6),
        "query_expansion_seconds": round(METRICS.query_expansion_seconds, 6),
        "cache_cpu_seconds": round(METRICS.cache_seconds, 6),
        "deduplication_seconds": round(METRICS.dedupe_seconds, 6),
        "ranking_total_seconds": round(METRICS.ranking_seconds, 6),
        "bi_encoder_seconds": round(
            METRICS.bi_encoder_query_seconds + METRICS.bi_encoder_papers_seconds, 6
        ),
        "cross_encoder_seconds": round(METRICS.cross_encoder_seconds, 6),
        "rrf_seconds": round(METRICS.rrf_seconds, 6),
        "replay_total_seconds": round(total, 6),
        "cache_miss_provider_calls": {provider.name: provider.calls for provider in providers},
        "result": {
            "candidate_count": response.candidate_count,
            "deduplicated_count": response.deduplicated_count,
            "ranked_count": response.ranked_count,
            "provider_status": response.provider_status,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
