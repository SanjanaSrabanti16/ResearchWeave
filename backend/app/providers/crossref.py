from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import httpx


class CrossrefError(RuntimeError):
    """A safe acquisition-only Crossref failure."""


@dataclass(frozen=True)
class CrossrefLocation:
    url: str
    is_pdf: bool


class CrossrefProvider:
    endpoint = "https://api.crossref.org/works"

    def __init__(self, client: httpx.AsyncClient, email: str | None = None) -> None:
        self.client = client
        self.email = email.strip() if email else None

    async def find_locations(self, doi: str) -> list[CrossrefLocation]:
        params = {"mailto": self.email} if self.email else None
        try:
            response = await self.client.get(
                f"{self.endpoint}/{quote(doi, safe='')}", params=params, timeout=15.0
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise CrossrefError("Crossref is temporarily unavailable") from exc
        if response.status_code == 404:
            return []
        if response.status_code >= 400:
            raise CrossrefError(f"Crossref rejected the request ({response.status_code})")
        try:
            payload = response.json()
            message = payload.get("message")
        except (ValueError, AttributeError) as exc:
            raise CrossrefError("Crossref returned an invalid response") from exc
        if not isinstance(message, dict):
            raise CrossrefError("Crossref returned an invalid response")
        locations: list[CrossrefLocation] = []
        links = message.get("link")
        if isinstance(links, list):
            for link in links:
                if not isinstance(link, dict):
                    continue
                url = link.get("URL")
                content_type = str(link.get("content-type", "")).casefold()
                if isinstance(url, str) and url.strip():
                    locations.append(CrossrefLocation(url.strip(), "pdf" in content_type))
        landing = message.get("URL")
        if isinstance(landing, str) and landing.strip():
            locations.append(CrossrefLocation(landing.strip(), False))
        return locations
