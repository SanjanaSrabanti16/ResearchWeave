from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class UnpaywallError(RuntimeError):
    """A safe error raised when Unpaywall cannot provide a usable response."""


class UnpaywallProvider:
    endpoint = "https://api.unpaywall.org/v2"

    def __init__(self, client: httpx.AsyncClient, email: str | None) -> None:
        self.client = client
        self.email = email.strip() if email else None

    @staticmethod
    def _pdf_url(payload: dict[str, Any]) -> str | None:
        if payload.get("is_oa") is not True:
            return None
        locations: list[Any] = [payload.get("best_oa_location")]
        other_locations = payload.get("oa_locations")
        if isinstance(other_locations, list):
            locations.extend(other_locations)
        for location in locations:
            if not isinstance(location, dict):
                continue
            url = location.get("url_for_pdf")
            if isinstance(url, str) and url.strip():
                return url.strip()
        return None

    async def find_pdf(self, doi: str) -> str | None:
        if not self.email:
            return None
        try:
            response = await self.client.get(
                f"{self.endpoint}/{quote(doi, safe='')}", params={"email": self.email}
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise UnpaywallError("Unpaywall is temporarily unavailable") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise UnpaywallError(f"Unpaywall rejected the request ({response.status_code})")
        try:
            payload = response.json()
        except ValueError as exc:
            raise UnpaywallError("Unpaywall returned an invalid response") from exc
        if not isinstance(payload, dict):
            raise UnpaywallError("Unpaywall returned an invalid response")
        return self._pdf_url(payload)
