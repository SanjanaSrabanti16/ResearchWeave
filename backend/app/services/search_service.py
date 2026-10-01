from __future__ import annotations

import asyncio
import logging
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
from app.services.provider_reliability import (
    TRANSIENT_FAILURE_CATEGORIES,
    CircuitBreaker,
    CircuitOpenError,
)
from app.services.query_variants import QueryVariantService
from app.services.ranking_service import RankingService

logger = logging.getLogger(__name__)


class AllProvidersFailedError(RuntimeError):
    pass


@dataclass
class ProviderResult:
    name: str
    papers: list[PaperCandidate]
    status: str
    warning: str | None = None
    category: str | None = None


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
                f"{provider_name} is busy; other provider results remain available",
                category="timeout",
                attempts=1,
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
        circuit_breaker: CircuitBreaker | None = None,
        overall_deadline_seconds: float = 35.0,
        min_variant_gain: int = 1,
    ) -> None:
        self.providers = providers
        self.cache = cache
        self.deduplicator = deduplicator
        self.ranker = ranker
        self.candidate_target = candidate_target
        self.query_variants = query_variants or QueryVariantService()
        self.pacer = pacer or ProviderPacer()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.overall_deadline_seconds = overall_deadline_seconds
        self.min_variant_gain = min_variant_gain

    async def _search_provider(
        self,
        provider: SearchProvider,
        request: SearchRequest,
        provider_limit: int,
        variant_index: int = 0,
    ) -> ProviderResult:
        started = time.monotonic()
        cache_warning: str | None = None
        operation_key = (provider.name, "search")
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
                logger.info(
                    "provider_variant_completed provider=%s operation=search variant=%s "
                    "outcome=cached duration_ms=%s",
                    provider.name,
                    variant_index,
                    round((time.monotonic() - started) * 1000),
                )
                return ProviderResult(provider.name, cached, "cached")
        except Exception:
            cache_warning = f"{provider.name} cache read failed; queried the provider directly"

        try:
            self.circuit_breaker.before_call(operation_key)
            papers = await self.pacer.run(
                provider.name,
                lambda: provider.search(
                    request.query, provider_limit, request.start_year, request.end_year
                ),
            )
        except CircuitOpenError:
            logger.info(
                "provider_skipped provider=%s operation=search variant=%s "
                "category=circuit_open circuit_state=open",
                provider.name,
                variant_index,
            )
            return ProviderResult(
                provider.name,
                [],
                "error",
                f"{provider.name} is temporarily unavailable",
                "circuit_open",
            )
        except ProviderError as exc:
            category = getattr(exc, "category", "invalid_response")
            self.circuit_breaker.record_failure(operation_key, category)
            logger.warning(
                "provider_variant_failed provider=%s operation=search variant=%s "
                "category=%s http_status=%s attempts=%s success=false duration_ms=%s "
                "circuit_state=%s",
                provider.name,
                variant_index,
                getattr(exc, "category", "provider_error"),
                getattr(exc, "status_code", None),
                getattr(exc, "attempts", None),
                round((time.monotonic() - started) * 1000),
                self.circuit_breaker.state(operation_key),
            )
            return ProviderResult(provider.name, [], "error", str(exc), category)
        except Exception:
            self.circuit_breaker.record_failure(operation_key, "network_error")
            return ProviderResult(
                provider.name,
                [],
                "error",
                f"{provider.name} failed unexpectedly; other provider results remain available",
                "network_error",
            )

        self.circuit_breaker.record_success(operation_key)
        logger.info(
            "provider_variant_completed provider=%s operation=search variant=%s "
            "outcome=success papers=%s duration_ms=%s circuit_state=%s",
            provider.name,
            variant_index,
            len(papers),
            round((time.monotonic() - started) * 1000),
            self.circuit_breaker.state(operation_key),
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
            unique_papers: list[PaperCandidate] = []
            for index, variant in enumerate(variants):
                result = await self._search_provider(
                    provider,
                    request.model_copy(update={"query": variant}),
                    original_limit if index == 0 else variant_limit,
                    index,
                )
                provider_results.append(result)
                if result.status == "error":
                    if result.category in TRANSIENT_FAILURE_CATEGORIES | {"circuit_open"}:
                        logger.info(
                            "provider_expansion_stopped provider=%s variant=%s reason=%s",
                            provider.name,
                            index,
                            result.category,
                        )
                        break
                    continue
                previous_count = len(unique_papers)
                unique_papers = self.deduplicator.deduplicate([*unique_papers, *result.papers])
                marginal_gain = len(unique_papers) - previous_count
                if len(unique_papers) >= original_limit:
                    logger.info(
                        "provider_expansion_stopped provider=%s variant=%s "
                        "reason=candidate_target unique=%s",
                        provider.name,
                        index,
                        len(unique_papers),
                    )
                    break
                if index > 0 and marginal_gain < self.min_variant_gain:
                    logger.info(
                        "provider_expansion_stopped provider=%s variant=%s "
                        "reason=low_marginal_gain gain=%s",
                        provider.name,
                        index,
                        marginal_gain,
                    )
                    break
            return provider_results

        started = time.monotonic()
        tasks = {
            asyncio.create_task(search_variants(provider)): provider for provider in self.providers
        }
        done, pending = await asyncio.wait(tasks, timeout=self.overall_deadline_seconds)
        completed = {tasks[task].name: task.result() for task in done}
        for task in pending:
            provider = tasks[task]
            self.circuit_breaker.record_failure((provider.name, "search"), "timeout")
            logger.warning(
                "provider_deadline_exceeded provider=%s operation=search category=timeout",
                provider.name,
            )
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        by_provider = [
            completed.get(
                provider.name,
                [
                    ProviderResult(
                        provider.name,
                        [],
                        "error",
                        f"{provider.name} exceeded the search deadline",
                        "timeout",
                    )
                ],
            )
            for provider in self.providers
        ]
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
        logger.info(
            "search_completed providers=%s variants=%s candidates=%s canonical=%s duration_ms=%s",
            len(self.providers),
            {
                provider.name: len(group)
                for provider, group in zip(self.providers, by_provider, strict=True)
            },
            len(candidates),
            len(deduplicated),
            round((time.monotonic() - started) * 1000),
        )
        ranked = await asyncio.to_thread(
            self.ranker.rank, request.query, deduplicated, request.limit
        )
        return SearchResponse(
            query=request.query,
            overall_status="success" if ranked else "no_results",
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
