from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.models.paper import PaperCandidate


class ProviderError(RuntimeError):
    """A safe, provider-scoped error for expected external failures."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_error",
        status_code: int | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.attempts = attempts

    @property
    def transient(self) -> bool:
        return self.category in {"timeout", "rate_limited", "server_error", "network_error"}


class SearchProvider(ABC):
    name: str

    def __init__(
        self,
        client: httpx.AsyncClient,
        retries: int = 2,
        retry_base_seconds: float = 0.5,
        min_retry_seconds: float = 0.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        self.client = client
        self.retries = retries
        self.retry_base_seconds = retry_base_seconds
        self.min_retry_seconds = min_retry_seconds
        self._sleep = sleep
        self._jitter = jitter or (lambda: random.uniform(0.0, 0.25))

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.get(url, **kwargs)
                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                if response.status_code >= 400:
                    raise ProviderError(
                        f"{self.name} rejected the request ({response.status_code})",
                        category=(
                            "authentication"
                            if response.status_code in {401, 403}
                            else "invalid_response"
                        ),
                        status_code=response.status_code,
                        attempts=attempt + 1,
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
                    delay = min(max(delay, self.min_retry_seconds), 10.0)
                    await (self._sleep or asyncio.sleep)(delay + self._jitter())
        status_code = (
            last_error.response.status_code
            if isinstance(last_error, httpx.HTTPStatusError)
            else None
        )
        if status_code == 429:
            category = "rate_limited"
        elif status_code in {408, 504} or isinstance(last_error, httpx.TimeoutException):
            category = "timeout"
        elif status_code in {500, 502, 503}:
            category = "server_error"
        else:
            category = "network_error"
        raise ProviderError(
            f"{self.name} is temporarily unavailable",
            category=category,
            status_code=status_code,
            attempts=self.retries + 1,
        ) from last_error

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[PaperCandidate]:
        raise NotImplementedError
