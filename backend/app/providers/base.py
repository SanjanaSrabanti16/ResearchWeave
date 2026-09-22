from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.models.paper import PaperCandidate


class ProviderError(RuntimeError):
    """A safe, provider-scoped error for expected external failures."""


class SearchProvider(ABC):
    name: str

    def __init__(
        self,
        client: httpx.AsyncClient,
        retries: int = 2,
        retry_base_seconds: float = 0.5,
        min_retry_seconds: float = 0.0,
    ) -> None:
        self.client = client
        self.retries = retries
        self.retry_base_seconds = retry_base_seconds
        self.min_retry_seconds = min_retry_seconds

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.get(url, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code >= 400:
                    raise ProviderError(
                        f"{self.name} rejected the request ({response.status_code})"
                    )
                return response
            except ProviderError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt < self.retries:
                    retry_after = None
                    if isinstance(exc, httpx.HTTPStatusError):
                        retry_after = exc.response.headers.get("retry-after")
                    try:
                        delay = (
                            float(retry_after)
                            if retry_after
                            else self.retry_base_seconds * 2**attempt
                        )
                    except ValueError:
                        delay = self.retry_base_seconds * 2**attempt
                    await asyncio.sleep(min(max(delay, self.min_retry_seconds), 10.0))
        raise ProviderError(f"{self.name} is temporarily unavailable") from last_error

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[PaperCandidate]:
        raise NotImplementedError
