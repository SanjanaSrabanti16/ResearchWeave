from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from app.models.api import ProviderHealth, SearchRequest, SearchResponse
from app.models.paper import PaperCandidate
from app.providers.base import ProviderError, SearchProvider
from app.services.cache_service import CacheService
from app.services.deduplication_service import DeduplicationService
from app.services.query_variants import QueryVariantService
from app.services.ranking_service import RankingService


class AllProvidersFailedError(RuntimeError):
    pass


@dataclass
class ProviderResult:
    name: str
    papers: list[PaperCandidate]
    status: str
    warning: str | None = None


_T = TypeVar("_T")


class ProviderPacer:
    """Space live provider requests across variants and concurrent searches."""

    def __init__(
        self,
        intervals: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_queue_seconds: float = 10.0,
    ) -> None:
        self.intervals = (
            intervals if intervals is not None else {"semantic_scholar": 1.0, "arxiv": 3.0}
        )
        self.clock = clock
        self.sleep = sleep
        self.max_queue_seconds = max_queue_seconds
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_started: dict[str, float] = {}

    async def run(self, provider_name: str, request: Callable[[], Awaitable[_T]]) -> _T:
        interval = self.intervals.get(provider_name, 0.0)
        if interval <= 0:
            return await request()
        lock = self._locks.setdefault(provider_name, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.max_queue_seconds)
        except TimeoutError as exc:
            raise ProviderError(
                f"{provider_name} is busy; other provider results remain available"
            ) from exc
        try:
            last_started = self._last_started.get(provider_name)
            if last_started is not None:
                delay = interval - (self.clock() - last_started)
                if delay > 0:
                    await self.sleep(delay)
            self._last_started[provider_name] = self.clock()
            return await request()
        finally:
            lock.release()


class SearchService:
    def __init__(
        self,
        providers: list[SearchProvider],
        cache: CacheService,
        deduplicator: DeduplicationService,
        ranker: RankingService,
        candidate_target: int = 240,
        query_variants: QueryVariantService | None = None,
        pacer: ProviderPacer | None = None,
    ) -> None:
        self.providers = providers
        self.cache = cache
        self.deduplicator = deduplicator
        self.ranker = ranker
        self.candidate_target = candidate_target
        self.query_variants = query_variants or QueryVariantService()
        self.pacer = pacer or ProviderPacer()

    async def _search_provider(
        self, provider: SearchProvider, request: SearchRequest, provider_limit: int
    ) -> ProviderResult:
        cache_warning: str | None = None
        try:
            cached = await asyncio.to_thread(
                self.cache.get,
                provider.name,
                request.query,
                request.start_year,
                request.end_year,
                provider_limit,
            )
            if cached is not None:
                return ProviderResult(provider.name, cached, "cached")
        except Exception:
            cache_warning = f"{provider.name} cache read failed; queried the provider directly"

        try:
            papers = await self.pacer.run(
                provider.name,
                lambda: provider.search(
                    request.query, provider_limit, request.start_year, request.end_year
                ),
            )
        except ProviderError as exc:
            return ProviderResult(provider.name, [], "error", str(exc))
        except Exception:
            return ProviderResult(
                provider.name,
                [],
                "error",
                f"{provider.name} failed unexpectedly; other provider results remain available",
            )

        try:
            await asyncio.to_thread(
                self.cache.set,
                provider.name,
                request.query,
                request.start_year,
                request.end_year,
                provider_limit,
                papers,
            )
        except Exception:
            cache_warning = f"{provider.name} results could not be cached"
        return ProviderResult(provider.name, papers, "ok", cache_warning)

    @staticmethod
    def _within_years(paper: PaperCandidate, start_year: int | None, end_year: int | None) -> bool:
        if paper.publication_year is None:
            return start_year is None and end_year is None
        if start_year is not None and paper.publication_year < start_year:
            return False
        return not (end_year is not None and paper.publication_year > end_year)

    @staticmethod
    def _provider_health(_name: str, results: list[ProviderResult]) -> ProviderHealth:
        successful = sum(result.status in {"ok", "cached"} for result in results)
        failed = sum(result.status == "error" for result in results)
        cached = sum(result.status == "cached" for result in results)
        if successful and not failed:
            status = "ok"
            message = None
        elif successful:
            status = "degraded"
            message = "Some query variants failed; partial results remain available."
        else:
            status = "unavailable"
            message = "No requests succeeded; this provider is temporarily unavailable."
        return ProviderHealth(
            status=status,
            successful_requests=successful,
            failed_requests=failed,
            cached_requests=cached,
            message=message,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        variants = self.query_variants.generate(request.query)
        original_limit = min(100, max(20, math.ceil(self.candidate_target / len(self.providers))))
        variant_limit = min(40, original_limit)

        async def search_variants(provider: SearchProvider) -> list[ProviderResult]:
            provider_results: list[ProviderResult] = []
            for index, variant in enumerate(variants):
                result = await self._search_provider(
                    provider,
                    request.model_copy(update={"query": variant}),
                    original_limit if index == 0 else variant_limit,
                )
                provider_results.append(result)
                if result.status == "error":
                    break
            return provider_results

        by_provider = await asyncio.gather(*(search_variants(p) for p in self.providers))
        results = [result for group in by_provider for result in group]
        if results and all(result.status == "error" for result in results):
            raise AllProvidersFailedError("All scholarly providers are currently unavailable.")

        candidates = [paper for result in results for paper in result.papers]
        filtered = [
            paper
            for paper in candidates
            if self._within_years(paper, request.start_year, request.end_year)
        ]
        deduplicated = self.deduplicator.deduplicate(filtered)
        ranked = await asyncio.to_thread(
            self.ranker.rank, request.query, deduplicated, request.limit
        )
        return SearchResponse(
            query=request.query,
            candidate_count=len(candidates),
            deduplicated_count=len(deduplicated),
            ranked_count=len(ranked),
            papers=ranked,
            provider_status={
                provider.name: self._provider_health(provider.name, group)
                for provider, group in zip(self.providers, by_provider, strict=True)
            },
            warnings=list(
                dict.fromkeys(
                    result.warning
                    for result in results
                    if result.warning and result.status != "error"
                )
            ),
        )
