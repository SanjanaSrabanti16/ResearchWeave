from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.models.api import SearchRequest
from app.models.paper import PaperCandidate
from app.providers.base import ProviderError
from app.services.deduplication_service import DeduplicationService
from app.services.provider_reliability import CircuitBreaker, CircuitOpenError, CircuitState
from app.services.search_service import SearchService


class _Cache:
    def get(self, *_: Any) -> None:
        return None

    def set(self, *_: Any) -> None:
        return None


class _Ranker:
    def rank(self, _query: str, papers: list[PaperCandidate], limit: int):
        return [paper.model_copy(update={"semantic_score": 0.5}) for paper in papers[:limit]]


class _Variants:
    def generate(self, query: str) -> list[str]:
        return [query, "variant one", "variant two", "variant three"]


class _Provider:
    def __init__(self, name: str, outcomes: list[object]) -> None:
        self.name = name
        self.outcomes = outcomes
        self.calls = 0

    async def search(self, *_: Any) -> list[PaperCandidate]:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


def _service(providers: list[_Provider], **kwargs: Any) -> SearchService:
    return SearchService(
        providers,  # type: ignore[arg-type]
        _Cache(),  # type: ignore[arg-type]
        DeduplicationService(),
        _Ranker(),  # type: ignore[arg-type]
        query_variants=_Variants(),  # type: ignore[arg-type]
        **kwargs,
    )


def test_circuit_breaker_open_half_open_close_and_independent_keys() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=10, clock=lambda: now[0])
    search_key = ("arxiv", "search")
    pdf_key = ("arxiv", "pdf")

    breaker.record_failure(search_key, "timeout")
    assert breaker.state(search_key) is CircuitState.CLOSED
    breaker.record_failure(search_key, "server_error")
    assert breaker.state(search_key) is CircuitState.OPEN
    assert breaker.state(pdf_key) is CircuitState.CLOSED
    with pytest.raises(CircuitOpenError):
        breaker.before_call(search_key)

    now[0] = 10.0
    assert breaker.before_call(search_key) is CircuitState.HALF_OPEN
    breaker.record_success(search_key)
    assert breaker.state(search_key) is CircuitState.CLOSED


def test_half_open_failure_reopens_and_non_transient_does_not_open() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=lambda: now[0])
    key = ("gemini", "generation")
    breaker.record_failure(key, "authentication")
    assert breaker.state(key) is CircuitState.CLOSED
    breaker.record_failure(key, "network_error")
    now[0] = 5.0
    breaker.before_call(key)
    breaker.record_failure(key, "timeout")
    assert breaker.state(key) is CircuitState.OPEN


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["rate_limited", "timeout", "server_error"])
async def test_first_transient_failure_stops_variants_and_sibling_continues(category: str) -> None:
    failed = _Provider("arxiv", [ProviderError("temporary", category=category)])
    healthy = _Provider(
        "openalex",
        [
            [PaperCandidate(title="Healthy result")],
            [],
        ],
    )
    response = await _service([failed, healthy]).search(SearchRequest(query="agent systems"))

    assert failed.calls == 1
    assert healthy.calls == 2
    assert response.papers[0].title == "Healthy result"
    assert response.provider_status["arxiv"].status == "unavailable"


@pytest.mark.asyncio
async def test_later_transient_failure_returns_degraded_partial_results() -> None:
    provider = _Provider(
        "semantic_scholar",
        [
            [PaperCandidate(title="First result")],
            ProviderError("temporary", category="server_error"),
        ],
    )
    response = await _service([provider]).search(SearchRequest(query="agent systems"))

    assert provider.calls == 2
    assert response.provider_status["semantic_scholar"].status == "degraded"
    assert response.papers[0].title == "First result"


@pytest.mark.asyncio
async def test_candidate_target_and_zero_gain_stop_expansion() -> None:
    target_papers = [PaperCandidate(title=f"Paper {index}") for index in range(30)]
    target = _Provider("openalex", [target_papers])
    await _service([target], candidate_target=30).search(SearchRequest(query="agent systems"))
    assert target.calls == 1

    duplicate = PaperCandidate(title="Same paper", doi="10.1/same")
    zero_gain = _Provider("semantic_scholar", [[duplicate], [duplicate]])
    await _service([zero_gain]).search(SearchRequest(query="agent systems"))
    assert zero_gain.calls == 2


@pytest.mark.asyncio
async def test_overall_deadline_returns_fast_provider_results() -> None:
    class _Hanging(_Provider):
        async def search(self, *_: Any) -> list[PaperCandidate]:
            self.calls += 1
            await asyncio.Event().wait()
            return []

    fast = _Provider("openalex", [[PaperCandidate(title="Fast result")], []])
    hanging = _Hanging("arxiv", [[]])
    response = await _service([fast, hanging], overall_deadline_seconds=0.2).search(
        SearchRequest(query="agent systems")
    )

    assert response.papers[0].title == "Fast result"
    assert response.provider_status["arxiv"].status == "unavailable"
    assert response.provider_status["arxiv"].message is not None
