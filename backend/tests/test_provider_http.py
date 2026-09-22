import httpx
import pytest

from app.providers.base import ProviderError
from app.providers.openalex import OpenAlexProvider
from app.providers.semantic_scholar import SemanticScholarProvider


@pytest.mark.asyncio
async def test_openalex_search_uses_documented_shape_and_date_filter() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "visual agents"
        assert request.url.params["per_page"] == "5"
        assert request.url.params["per_page"] == "5"
        assert "per-page" not in request.url.params
        assert request.url.params["api_key"] == "test-key"
        assert request.url.params["filter"] == (
            "from_publication_date:2020-01-01,to_publication_date:2025-12-31"
        )
        return httpx.Response(200, json={"results": [{"id": "W1", "title": "A result"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        papers = await OpenAlexProvider(client, api_key="test-key").search(
            "visual agents", 5, 2020, 2025
        )
    assert [paper.title for paper in papers] == ["A result"]


@pytest.mark.asyncio
async def test_openalex_uses_optional_api_key_and_caps_request_at_100() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "test-key"
        assert request.url.params["per_page"] == "100"
        assert "mailto" not in request.url.params
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await OpenAlexProvider(client, api_key="test-key").search("agents", 200)


@pytest.mark.asyncio
async def test_semantic_scholar_search_uses_data_array_and_year_filter() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["year"] == "2020-2025"
        assert "externalIds" in request.url.params["fields"]
        return httpx.Response(
            200,
            json={"data": [{"paperId": "S1", "title": "Semantic result", "year": 2024}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        papers = await SemanticScholarProvider(client).search("visual agents", 5, 2020, 2025)
    assert papers[0].semantic_scholar_id == "S1"


@pytest.mark.asyncio
async def test_semantic_scholar_retry_waits_at_least_one_second(monkeypatch) -> None:
    calls = 0
    delays: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json={"data": []})

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("app.providers.base.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await SemanticScholarProvider(client, retries=1).search("agents", 5)
    assert calls == 2
    assert delays == [1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"not-json", b"[]", b'{"results": {}}'])
async def test_openalex_rejects_invalid_json_or_result_shape(payload: bytes) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError, match="invalid"):
            await OpenAlexProvider(client).search("agents", 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"not-json", b"[]", b'{"data": {}}'])
async def test_semantic_scholar_rejects_invalid_json_or_result_shape(payload: bytes) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError, match="invalid"):
            await SemanticScholarProvider(client).search("agents", 5)
