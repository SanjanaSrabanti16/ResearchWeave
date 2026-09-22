from typing import Any

import pytest

from app.models.api import SearchRequest
from app.models.paper import PaperCandidate
from app.providers.base import ProviderError
from app.services.deduplication_service import DeduplicationService
from app.services.search_service import AllProvidersFailedError, ProviderPacer, SearchService


class MemoryCache:
    def get(self, *_: Any) -> None:
        return None

    def set(self, *_: Any) -> None:
        return None


class ReadFailureCache(MemoryCache):
    def get(self, *_: Any) -> None:
        raise RuntimeError("cache read failed")


class WriteFailureCache(MemoryCache):
    def set(self, *_: Any) -> None:
        raise RuntimeError("cache write failed")


class RecordingCache(MemoryCache):
    def __init__(self) -> None:
        self.entries: dict[tuple[Any, ...], list[PaperCandidate]] = {}

    def get(self, *key: Any) -> list[PaperCandidate] | None:
        return self.entries.get(key)

    def set(self, *args: Any) -> None:
        *key, papers = args
        self.entries[tuple(key)] = papers


class StubProvider:
    def __init__(self, name: str, papers: list[PaperCandidate] | None = None, fail: bool = False):
        self.name = name
        self.papers = papers or []
        self.fail = fail

    async def search(self, *_: Any) -> list[PaperCandidate]:
        if self.fail:
            raise ProviderError(f"{self.name} unavailable")
        return self.papers


class StubRanker:
    def rank(self, query: str, papers: list[PaperCandidate], limit: int) -> list[PaperCandidate]:
        for index, paper in enumerate(papers):
            paper.semantic_score = 0.5
            paper.reranker_score = 1.0 - index / 10
        return papers[:limit]


@pytest.mark.asyncio
async def test_partial_provider_failure_and_year_filtering() -> None:
    providers = [
        StubProvider(
            "good",
            [
                PaperCandidate(title="In range", publication_year=2023, source_names=["good"]),
                PaperCandidate(title="Too old", publication_year=2018, source_names=["good"]),
                PaperCandidate(title="Unknown year", source_names=["good"]),
            ],
        ),
        StubProvider("bad", fail=True),
    ]
    service = SearchService(
        providers, MemoryCache(), DeduplicationService(), StubRanker(), candidate_target=20
    )
    response = await service.search(SearchRequest(query="agents", start_year=2020, limit=5))
    assert response.candidate_count == 6
    assert response.deduplicated_count == 1
    assert response.provider_status == {"good": "ok", "bad": "error"}
    assert response.warnings == ["bad unavailable"]
    assert [paper.title for paper in response.papers] == ["In range"]


@pytest.mark.parametrize(
    ("start_year", "end_year", "expected"),
    [(None, None, True), (2020, None, False), (None, 2026, False), (2020, 2026, False)],
)
def test_unknown_year_filtering_is_strict_when_any_bound_is_set(
    start_year: int | None, end_year: int | None, expected: bool
) -> None:
    paper = PaperCandidate(title="Unknown year")
    assert SearchService._within_years(paper, start_year, end_year) is expected


@pytest.mark.parametrize(
    ("start_year", "end_year", "paper_year", "expected"),
    [
        (2020, None, 2019, False),
        (2020, None, 2020, True),
        (None, 2026, 2027, False),
        (None, 2026, 2026, True),
        (2020, 2026, 2019, False),
        (2020, 2026, 2020, True),
        (2020, 2026, 2026, True),
        (2020, 2026, 2027, False),
    ],
)
def test_known_year_filtering_is_inclusive_and_respects_each_bound(
    start_year: int | None,
    end_year: int | None,
    paper_year: int,
    expected: bool,
) -> None:
    paper = PaperCandidate(title="Known year", publication_year=paper_year)
    assert SearchService._within_years(paper, start_year, end_year) is expected


@pytest.mark.asyncio
async def test_all_providers_failing_raises_service_error() -> None:
    service = SearchService(
        [StubProvider("first", fail=True), StubProvider("second", fail=True)],
        MemoryCache(),
        DeduplicationService(),
        StubRanker(),
    )
    with pytest.raises(AllProvidersFailedError, match="All scholarly providers"):
        await service.search(SearchRequest(query="agents", limit=5))


@pytest.mark.asyncio
async def test_cache_read_failure_warns_and_uses_provider() -> None:
    service = SearchService(
        [StubProvider("source", [PaperCandidate(title="Result")])],
        ReadFailureCache(),
        DeduplicationService(),
        StubRanker(),
    )
    response = await service.search(SearchRequest(query="agents", limit=5))
    assert response.ranked_count == 1
    assert response.warnings == ["source cache read failed; queried the provider directly"]


@pytest.mark.asyncio
async def test_cache_write_failure_warns_and_preserves_results() -> None:
    service = SearchService(
        [StubProvider("source", [PaperCandidate(title="Result")])],
        WriteFailureCache(),
        DeduplicationService(),
        StubRanker(),
    )
    response = await service.search(SearchRequest(query="agents", limit=5))
    assert response.ranked_count == 1
    assert response.warnings == ["source results could not be cached"]


class RecordingProvider:
    def __init__(self, name: str, papers_by_query: dict[str, list[PaperCandidate]]) -> None:
        self.name = name
        self.papers_by_query = papers_by_query
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int, *_: Any) -> list[PaperCandidate]:
        self.calls.append((query, limit))
        return self.papers_by_query.get(query, [])


@pytest.mark.asyncio
async def test_variant_results_are_unioned_then_canonically_deduplicated() -> None:
    shared_a = PaperCandidate(
        title="Shared paper", doi="10.1234/shared", publication_year=2025, source_names=["first"]
    )
    shared_b = PaperCandidate(
        title="Shared paper", doi="10.1234/shared", publication_year=2025, source_names=["second"]
    )
    review = PaperCandidate(title="New paper", publication_year=2025, source_names=["first"])
    first = RecordingProvider(
        "first",
        {"AI agents for visualizations": [shared_a], "AI visualization": [review]},
    )
    second = RecordingProvider("second", {"AI agent visualization": [shared_b]})
    service = SearchService([first, second], MemoryCache(), DeduplicationService(), StubRanker())

    response = await service.search(
        SearchRequest(query="AI agents for visualizations", start_year=2025, end_year=2026)
    )

    assert response.candidate_count == 3
    assert response.deduplicated_count == 2
    assert {paper.title for paper in response.papers} == {"Shared paper", "New paper"}
    shared = next(p for p in response.papers if p.title == "Shared paper")
    assert shared.source_names == ["first", "second"]
    assert len(first.calls) == len(second.calls) == 4
    assert first.calls[0][1] == 100
    assert all(limit == 40 for _, limit in first.calls[1:])


@pytest.mark.asyncio
async def test_each_variant_uses_normal_provider_cache_key() -> None:
    provider = RecordingProvider(
        "openalex",
        {"AI agents for visualizations": [PaperCandidate(title="Result", publication_year=2025)]},
    )
    cache = RecordingCache()
    service = SearchService([provider], cache, DeduplicationService(), StubRanker())
    request = SearchRequest(query="AI agents for visualizations", start_year=2025, end_year=2026)

    first = await service.search(request)
    second = await service.search(request)

    assert first.provider_status == {"openalex": "ok"}
    assert second.provider_status == {"openalex": "cached"}
    assert len(provider.calls) == 4
    assert len(cache.entries) == 4


@pytest.mark.asyncio
async def test_provider_pacing_spaces_semantic_scholar_and_arxiv_requests() -> None:
    now = [0.0]
    delays: list[float] = []
    starts: list[tuple[str, float]] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)
        now[0] += seconds

    async def request(name: str) -> None:
        starts.append((name, now[0]))

    pacer = ProviderPacer(clock=lambda: now[0], sleep=fake_sleep)
    await pacer.run("semantic_scholar", lambda: request("semantic_scholar"))
    await pacer.run("semantic_scholar", lambda: request("semantic_scholar"))
    await pacer.run("arxiv", lambda: request("arxiv"))
    await pacer.run("arxiv", lambda: request("arxiv"))

    assert starts == [
        ("semantic_scholar", 0.0),
        ("semantic_scholar", 1.0),
        ("arxiv", 1.0),
        ("arxiv", 4.0),
    ]
    assert delays == [1.0, 3.0]
