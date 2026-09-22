from __future__ import annotations

from typing import Any

from app.models.paper import PaperCandidate
from app.providers.base import ProviderError, SearchProvider
from app.services.normalization import clean_text, normalize_doi, stable_paper_id


class OpenAlexProvider(SearchProvider):
    name = "openalex"
    endpoint = "https://api.openalex.org/works"

    def __init__(self, *args: Any, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.api_key = api_key

    @staticmethod
    def _abstract(inverted_index: Any) -> str | None:
        if not isinstance(inverted_index, dict):
            return None
        positioned: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            if isinstance(word, str) and isinstance(positions, list):
                positioned.extend(
                    (position, word) for position in positions if isinstance(position, int)
                )
        if not positioned:
            return None
        return " ".join(word for _, word in sorted(positioned))

    @classmethod
    def normalize(cls, item: dict[str, Any]) -> PaperCandidate | None:
        title = clean_text(item.get("display_name") or item.get("title"))
        if not title:
            return None
        ids = item.get("ids") if isinstance(item.get("ids"), dict) else {}
        doi = normalize_doi(ids.get("doi") or item.get("doi"))
        openalex_id = clean_text(ids.get("openalex") or item.get("id"))
        authorships = item.get("authorships") if isinstance(item.get("authorships"), list) else []
        authors = [
            name
            for row in authorships
            if isinstance(row, dict)
            and isinstance(row.get("author"), dict)
            and (name := clean_text(row["author"].get("display_name")))
        ]
        primary = (
            item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
        )
        source = primary.get("source") if isinstance(primary.get("source"), dict) else {}
        best_oa = (
            item.get("best_oa_location") if isinstance(item.get("best_oa_location"), dict) else {}
        )
        year = item.get("publication_year")
        if not isinstance(year, int):
            year = None
        citations = item.get("cited_by_count")
        if not isinstance(citations, int) or citations < 0:
            citations = None
        return PaperCandidate(
            id=stable_paper_id(doi=doi, arxiv_id=None, title=title),
            title=title,
            abstract=cls._abstract(item.get("abstract_inverted_index")),
            authors=authors,
            publication_year=year,
            publication_date=clean_text(item.get("publication_date")),
            venue=clean_text(source.get("display_name")),
            doi=doi,
            openalex_id=openalex_id,
            url=clean_text(primary.get("landing_page_url") or openalex_id),
            pdf_url=clean_text(best_oa.get("pdf_url") or primary.get("pdf_url")),
            citation_count=citations,
            source_names=[cls.name],
        )

    async def search(
        self,
        query: str,
        limit: int,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[PaperCandidate]:
        filters: list[str] = []
        if start_year:
            filters.append(f"from_publication_date:{start_year}-01-01")
        if end_year:
            filters.append(f"to_publication_date:{end_year}-12-31")
        params: dict[str, str | int] = {"search": query, "per_page": min(limit, 100)}
        if filters:
            params["filter"] = ",".join(filters)
        if self.api_key:
            params["api_key"] = self.api_key
        response = await self._get(self.endpoint, params=params)
        try:
            results = response.json().get("results", [])
        except (ValueError, AttributeError) as exc:
            raise ProviderError("openalex returned an invalid response") from exc
        if not isinstance(results, list):
            raise ProviderError("openalex returned an invalid result set")
        papers = [
            paper for item in results if isinstance(item, dict) if (paper := self.normalize(item))
        ]
        return papers[:limit]
