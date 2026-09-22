import httpx
import pytest

from app.models.paper import PaperCandidate
from app.providers.base import ProviderError, SearchProvider


class ProbeProvider(SearchProvider):
    name = "probe"

    async def search(
        self,
        query: str,
        limit: int,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[PaperCandidate]:
        del query, limit, start_year, end_year
        return []

    async def request(self) -> httpx.Response:
        return await self._get("https://provider.test/search")


@pytest.mark.asyncio
async def test_429_retries_and_honors_numeric_retry_after(monkeypatch) -> None:
    calls = 0
    sleep_delays: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200)

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr("app.providers.base.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await ProbeProvider(client, retries=1).request()
    assert response.status_code == 200
    assert calls == 2
    assert sleep_delays == [0.25]


@pytest.mark.asyncio
async def test_5xx_retries_without_real_sleep(monkeypatch) -> None:
    calls = 0
    sleep_delays: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200)

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr("app.providers.base.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await ProbeProvider(client, retries=1).request()
    assert response.status_code == 200
    assert calls == 2
    assert sleep_delays == [0.5]


@pytest.mark.asyncio
async def test_timeout_retries_without_real_sleep(monkeypatch) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200)

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("app.providers.base.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await ProbeProvider(client, retries=1).request()
    assert response.status_code == 200
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_provider_error(monkeypatch) -> None:
    calls = 0
    sleep_delays: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr("app.providers.base.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError, match="temporarily unavailable"):
            await ProbeProvider(client, retries=2).request()
    assert calls == 3
    assert sleep_delays == [0.5, 1.0]
