from __future__ import annotations

from typing import Any

from app.models.paper import PaperCandidate
from app.providers.base import ProviderError, SearchProvider
from app.services.normalization import (
    clean_text,
    normalize_arxiv_id,
    normalize_doi,
    stable_paper_id,
)


class SemanticScholarProvider(SearchProvider):
    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = (
        "title,abstract,authors,year,publicationDate,venue,externalIds,url,"
        "openAccessPdf,citationCount"
    )

    def __init__(self, *args: Any, api_key: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("min_retry_seconds", 1.0)
        super().__init__(*args, **kwargs)
        self.api_key = api_key

    @classmethod
    def normalize(cls, item: dict[str, Any]) -> PaperCandidate | None:
        title = clean_text(item.get("title"))
        if not title:
            return None
        external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        doi = normalize_doi(external.get("DOI"))
        arxiv_id = normalize_arxiv_id(external.get("ArXiv"))
        author_rows = item.get("authors") if isinstance(item.get("authors"), list) else []
        authors = [
            name
            for author in author_rows
            if isinstance(author, dict) and (name := clean_text(author.get("name")))
        ]
        pdf = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
        year = item.get("year") if isinstance(item.get("year"), int) else None
        citations = item.get("citationCount")
        if not isinstance(citations, int) or citations < 0:
            citations = None
        return PaperCandidate(
            id=stable_paper_id(doi=doi, arxiv_id=arxiv_id, title=title),
            title=title,
            abstract=clean_text(item.get("abstract")),
            authors=authors,
            publication_year=year,
            publication_date=clean_text(item.get("publicationDate")),
            venue=clean_text(item.get("venue")),
            doi=doi,
            arxiv_id=arxiv_id,
            semantic_scholar_id=clean_text(item.get("paperId")),
            url=clean_text(item.get("url")),
            pdf_url=clean_text(pdf.get("url")),
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
        params: dict[str, str | int] = {
            "query": query.replace("-", " "),
            "limit": min(limit, 100),
            "fields": self.fields,
        }
        if start_year or end_year:
            params["year"] = f"{start_year or ''}-{end_year or ''}"
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        response = await self._get(self.endpoint, params=params, headers=headers)
        try:
            results = response.json().get("data", [])
        except (ValueError, AttributeError) as exc:
            raise ProviderError("semantic_scholar returned an invalid response") from exc
        if not isinstance(results, list):
            raise ProviderError("semantic_scholar returned an invalid result set")
        papers = [
            paper for item in results if isinstance(item, dict) if (paper := self.normalize(item))
        ]
        return papers[:limit]
