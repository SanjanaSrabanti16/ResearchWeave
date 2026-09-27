import threading
from datetime import UTC, datetime

import arxiv
import httpx
import pytest

from app.providers.arxiv import ArxivProvider, _TimeoutSession
from app.providers.base import ProviderError


class FakeArxivClient:
    def __init__(self, results: list[arxiv.Result] | None = None, error: Exception | None = None):
        self.items = results or []
        self.error = error
        self.searches: list[arxiv.Search] = []
        self.thread_id: int | None = None

    def results(self, search: arxiv.Search):
        self.searches.append(search)
        self.thread_id = threading.get_ident()
        if self.error is not None:
            raise self.error
        return iter(self.items)


def _result(year: int) -> arxiv.Result:
    return arxiv.Result(
        entry_id=f"https://arxiv.org/abs/{year}01.01234",
        published=datetime(year, 1, 2, tzinfo=UTC),
        title=f"Visual agents {year}",
        summary="An abstract",
    )


@pytest.mark.asyncio
async def test_arxiv_client_has_safe_paging_rate_limit_and_retries() -> None:
    async with httpx.AsyncClient() as async_client:
        provider = ArxivProvider(async_client)
        assert provider.arxiv_client.page_size <= 100
        assert provider.arxiv_client.delay_seconds >= 3
        assert provider.arxiv_client.num_retries == 2
        assert isinstance(provider.arxiv_client._session, _TimeoutSession)
        assert provider.arxiv_client._session.timeout_seconds == 10.0


def test_arxiv_timeout_session_applies_default_and_preserves_explicit_timeout() -> None:
    class RecordingSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def get(self, _: str, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return object()

    delegate = RecordingSession()
    session = _TimeoutSession(delegate, 7.5)

    session.get("https://export.arxiv.org/api/query")
    session.get("https://export.arxiv.org/api/query", timeout=2.0)

    assert delegate.calls == [{"timeout": 7.5}, {"timeout": 2.0}]


@pytest.mark.asyncio
async def test_arxiv_request_timeout_must_be_positive() -> None:
    async with httpx.AsyncClient() as async_client:
        with pytest.raises(ValueError, match="timeouts must be positive"):
            ArxivProvider(async_client, request_timeout_seconds=0)


@pytest.mark.asyncio
async def test_busy_arxiv_provider_fails_without_waiting_for_locked_client() -> None:
    fake_client = FakeArxivClient([_result(2025)])
    async with httpx.AsyncClient() as async_client:
        provider = ArxivProvider(
            async_client,
            arxiv_client=fake_client,
            lock_timeout_seconds=0.01,
        )
        await provider._search_lock.acquire()
        try:
            with pytest.raises(ProviderError, match="busy"):
                await provider.search("visual agents", 20)
        finally:
            provider._search_lock.release()

    assert fake_client.searches == []


@pytest.mark.asyncio
async def test_arxiv_search_uses_package_in_worker_thread_and_filters_years() -> None:
    fake_client = FakeArxivClient([_result(year) for year in (2024, 2025, 2026, 2027)])
    main_thread_id = threading.get_ident()

    async with httpx.AsyncClient() as async_client:
        provider = ArxivProvider(async_client, arxiv_client=fake_client)
        papers = await provider.search("AI agents for visualizations", 80, 2025, 2026)

    assert [paper.publication_year for paper in papers] == [2025, 2026]
    assert all(paper.source_names == ["arxiv"] for paper in papers)
    assert fake_client.thread_id != main_thread_id
    assert len(fake_client.searches) == 1
    search = fake_client.searches[0]
    assert search.query == "all:AI AND all:agents AND all:visualizations"
    assert search.max_results == 80
    assert search.sort_by == arxiv.SortCriterion.Relevance
    assert search.sort_order == arxiv.SortOrder.Descending


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        arxiv.HTTPError("https://export.arxiv.org/api/query", 5, 406),
        ConnectionError("network failed"),
    ],
)
async def test_arxiv_library_failures_become_provider_errors(failure: Exception) -> None:
    fake_client = FakeArxivClient(error=failure)

    async with httpx.AsyncClient() as async_client:
        provider = ArxivProvider(async_client, arxiv_client=fake_client)
        with pytest.raises(ProviderError, match="arxiv is temporarily unavailable") as error:
            await provider.search("visual agents", 20)

    assert error.value.__cause__ is failure
