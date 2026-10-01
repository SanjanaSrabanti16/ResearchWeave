from __future__ import annotations

from urllib.parse import quote

import httpx

from app.services.normalization import normalize_pmcid


class EuropePMCError(RuntimeError):
    """A safe acquisition-only Europe PMC failure."""


class EuropePMCProvider:
    endpoint = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def resolve_pmcids(self, doi: str | None, known_values: list[str]) -> list[str]:
        pmcids = list(
            dict.fromkeys(pmcid for value in known_values if (pmcid := normalize_pmcid(value)))
        )
        if pmcids or not doi:
            return pmcids
        try:
            response = await self.client.get(
                self.endpoint,
                params={"query": f'DOI:"{doi}"', "format": "json", "pageSize": 3},
                timeout=15.0,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise EuropePMCError("Europe PMC is temporarily unavailable") from exc
        if response.status_code >= 400:
            raise EuropePMCError(f"Europe PMC rejected the request ({response.status_code})")
        try:
            rows = response.json().get("resultList", {}).get("result", [])
        except (ValueError, AttributeError) as exc:
            raise EuropePMCError("Europe PMC returned an invalid response") from exc
        if not isinstance(rows, list):
            raise EuropePMCError("Europe PMC returned an invalid response")
        for row in rows:
            if isinstance(row, dict) and (pmcid := normalize_pmcid(row.get("pmcid"))):
                pmcids.append(pmcid)
        return list(dict.fromkeys(pmcids))

    @staticmethod
    def pdf_locations(pmcid: str) -> list[str]:
        safe = quote(pmcid, safe="")
        return [
            f"https://europepmc.org/articles/{safe}?pdf=render",
            f"https://pmc.ncbi.nlm.nih.gov/articles/{safe}/pdf/",
        ]
