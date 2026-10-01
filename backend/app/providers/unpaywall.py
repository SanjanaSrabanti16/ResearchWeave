from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


class UnpaywallError(RuntimeError):
    """A safe error raised when Unpaywall cannot provide a usable response."""


@dataclass(frozen=True)
class UnpaywallLocation:
    url: str
    is_pdf: bool
    host_type: str | None = None


class UnpaywallProvider:
    endpoint = "https://api.unpaywall.org/v2"

    def __init__(self, client: httpx.AsyncClient, email: str | None) -> None:
        self.client = client
        self.email = email.strip() if email else None

    @staticmethod
    def _locations(payload: dict[str, Any]) -> list[UnpaywallLocation]:
        if payload.get("is_oa") is not True:
            return []
        locations: list[Any] = [payload.get("best_oa_location")]
        other_locations = payload.get("oa_locations")
        if isinstance(other_locations, list):
            locations.extend(other_locations)
        result: list[UnpaywallLocation] = []
        seen: set[str] = set()
        for location in locations:
            if not isinstance(location, dict):
                continue
            for key, is_pdf in (("url_for_pdf", True), ("url", False), ("landing_page_url", False)):
                url = location.get(key)
                if isinstance(url, str) and url.strip() and url.strip().casefold() not in seen:
                    seen.add(url.strip().casefold())
                    result.append(
                        UnpaywallLocation(
                            url.strip(), is_pdf, str(location.get("host_type") or "") or None
                        )
                    )
        return result

    @classmethod
    def _pdf_url(cls, payload: dict[str, Any]) -> str | None:
        return next((location.url for location in cls._locations(payload) if location.is_pdf), None)

    async def find_pdf(self, doi: str) -> str | None:
        return next(
            (location.url for location in await self.find_locations(doi) if location.is_pdf),
            None,
        )

    async def find_locations(self, doi: str) -> list[UnpaywallLocation]:
        if not self.email:
            return []
        try:
            response = await self.client.get(
                f"{self.endpoint}/{quote(doi, safe='')}", params={"email": self.email}
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise UnpaywallError("Unpaywall is temporarily unavailable") from exc
        if response.status_code == 404:
            return []
        if response.status_code >= 400:
            raise UnpaywallError(f"Unpaywall rejected the request ({response.status_code})")
        try:
            payload = response.json()
        except ValueError as exc:
            raise UnpaywallError("Unpaywall returned an invalid response") from exc
        if not isinstance(payload, dict):
            raise UnpaywallError("Unpaywall returned an invalid response")
        return self._locations(payload)
