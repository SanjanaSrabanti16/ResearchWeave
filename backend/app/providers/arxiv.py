from __future__ import annotations

import asyncio
import re

import arxiv
import httpx

from app.models.paper import PaperCandidate
from app.providers.base import ProviderError, SearchProvider
from app.services.normalization import (
    clean_text,
    normalize_arxiv_id,
    normalize_doi,
    normalize_metadata_text,
    stable_paper_id,
)

STOP_WORDS = {"a", "an", "and", "are", "for", "how", "in", "of", "on", "the", "to"}
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0


class _TimeoutSession:
    """Add a bounded timeout to the requests session owned by the arxiv package."""

    def __init__(self, session: object, timeout_seconds: float) -> None:
        self.session = session
        self.timeout_seconds = timeout_seconds

    def get(self, url: str, **kwargs: object):
        kwargs.setdefault("timeout", self.timeout_seconds)
        return self.session.get(url, **kwargs)


class ArxivProvider(SearchProvider):
    name = "arxiv"

    def __init__(
        self,
        client: httpx.AsyncClient,
        retries: int = 2,
        retry_base_seconds: float = 0.5,
        *,
        arxiv_client: arxiv.Client | None = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        lock_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(client, retries, retry_base_seconds)
        self.arxiv_client = arxiv_client or arxiv.Client(
            page_size=20, delay_seconds=3.0, num_retries=retries
        )
        if request_timeout_seconds <= 0 or lock_timeout_seconds <= 0:
            raise ValueError("arXiv timeouts must be positive")
        session = getattr(self.arxiv_client, "_session", None)
        if session is not None and not isinstance(session, _TimeoutSession):
            self.arxiv_client._session = _TimeoutSession(session, request_timeout_seconds)
        self._search_lock = asyncio.Lock()
        self.lock_timeout_seconds = lock_timeout_seconds

    @classmethod
    def normalize(cls, result: arxiv.Result) -> PaperCandidate | None:
        title = normalize_metadata_text(result.title)
        if not title:
            return None
        url = clean_text(result.entry_id)
        arxiv_id = normalize_arxiv_id(url)
        doi = normalize_doi(result.doi)
        published = result.published
        authors = [name for author in result.authors if (name := clean_text(author.name))]
        return PaperCandidate(
            id=stable_paper_id(doi=doi, arxiv_id=arxiv_id, title=title),
            title=title,
            abstract=normalize_metadata_text(result.summary),
            authors=authors,
            publication_year=published.year,
            publication_date=published.date().isoformat(),
            venue=clean_text(result.journal_ref) or "arXiv",
            doi=doi,
            arxiv_id=arxiv_id,
            url=url,
            pdf_url=clean_text(result.pdf_url),
            source_names=[cls.name],
        )

    async def search(
        self,
        query: str,
        limit: int,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[PaperCandidate]:
        safe_query = re.sub(r"[^\w\s-]", " ", query, flags=re.UNICODE)
        terms = [
            term
            for term in (clean_text(safe_query) or query).split()
            if term.casefold() not in STOP_WORDS
        ]
        search_query = " AND ".join(f"all:{term}" for term in terms) or f'all:"{query}"'
        search = arxiv.Search(
            query=search_query,
            max_results=min(limit, 100),
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            try:
                await asyncio.wait_for(
                    self._search_lock.acquire(), timeout=self.lock_timeout_seconds
                )
            except TimeoutError as exc:
                raise ProviderError(
                    "arxiv is busy; other provider results remain available"
                ) from exc
            try:
                results = await asyncio.to_thread(lambda: list(self.arxiv_client.results(search)))
            finally:
                self._search_lock.release()
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError("arxiv is temporarily unavailable") from exc
        papers: list[PaperCandidate] = []
        for result in results:
            paper = self.normalize(result)
            if paper is None:
                continue
            if start_year is not None and paper.publication_year < start_year:
                continue
            if end_year is not None and paper.publication_year > end_year:
                continue
            papers.append(paper)
        return papers[:limit]
