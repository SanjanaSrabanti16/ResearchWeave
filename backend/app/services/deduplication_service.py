from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

from app.models.paper import Paper, PaperCandidate
from app.services.normalization import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_metadata_text,
    normalize_title,
    stable_paper_id,
)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _url_key(value: str) -> str:
    parsed = urlsplit(value.strip())
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return value.strip().casefold()
    netloc = hostname if port in {None, 443} else f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold(), netloc, path, parsed.query, ""))


def _unique_urls(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or not value.strip():
            continue
        key = _url_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def _arxiv_identities(paper: PaperCandidate) -> list[str]:
    candidates = (
        [paper.arxiv_id]
        + paper.arxiv_ids
        + [paper.doi]
        + ([paper.url] if paper.url else [])
        + paper.alternate_urls
        + ([paper.pdf_url] if paper.pdf_url else [])
        + paper.alternate_pdf_urls
    )
    return _unique(
        identity for candidate in candidates if (identity := normalize_arxiv_id(candidate))
    )


class DeduplicationService:
    def deduplicate(self, papers: list[PaperCandidate]) -> list[Paper]:
        if not papers:
            return []

        parents = list(range(len(papers)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        doi_indexes: dict[str, int] = {}
        for index, paper in enumerate(papers):
            if doi := normalize_doi(paper.doi):
                if doi in doi_indexes:
                    union(index, doi_indexes[doi])
                else:
                    doi_indexes[doi] = index

        arxiv_indexes: dict[str, int] = {}
        for index, paper in enumerate(papers):
            for arxiv_id in _arxiv_identities(paper):
                if arxiv_id in arxiv_indexes:
                    union(index, arxiv_indexes[arxiv_id])
                else:
                    arxiv_indexes[arxiv_id] = index

        title_indexes: dict[str, list[int]] = defaultdict(list)
        for index, paper in enumerate(papers):
            if title := normalize_title(normalize_metadata_text(paper.title)):
                title_indexes[title].append(index)

        for matching_indexes in title_indexes.values():
            roots = list(dict.fromkeys(find(index) for index in matching_indexes))
            if len(roots) < 2:
                continue
            members_by_root: dict[int, list[PaperCandidate]] = defaultdict(list)
            for index, paper in enumerate(papers):
                root = find(index)
                if root in roots:
                    members_by_root[root].append(paper)
            dois = {
                doi
                for root in roots
                for paper in members_by_root[root]
                if (doi := normalize_doi(paper.doi)) and normalize_arxiv_id(doi) is None
            }
            arxiv_ids = {
                arxiv_id
                for root in roots
                for paper in members_by_root[root]
                for arxiv_id in _arxiv_identities(paper)
            }
            # An identifier-free record must never bridge distinct strong-ID groups.
            if len(dois) > 1 or len(arxiv_ids) > 1:
                continue
            for root in roots[1:]:
                union(roots[0], root)

        groups: dict[int, list[PaperCandidate]] = defaultdict(list)
        for index, paper in enumerate(papers):
            groups[find(index)].append(paper)
        return [self._merge(group) for group in groups.values()]

    @staticmethod
    def _merge(group: list[PaperCandidate]) -> Paper:
        def completeness(paper: PaperCandidate) -> tuple[int, int, int, int, int]:
            populated = sum(
                value is not None
                for value in (
                    paper.abstract,
                    paper.publication_year,
                    paper.publication_date,
                    paper.venue,
                    paper.doi,
                    paper.arxiv_id,
                    paper.url,
                    paper.pdf_url,
                )
            )
            publisher_venue = bool(
                paper.venue and normalize_title(paper.venue) not in {"arxiv", "arxiv org"}
            )
            return (
                int(normalize_doi(paper.doi) is not None),
                int(publisher_venue),
                populated,
                len(paper.abstract or ""),
                len(paper.authors),
            )

        base = max(group, key=completeness)

        def first(attribute: str) -> object | None:
            for paper in [base, *group]:
                value = getattr(paper, attribute)
                if value is not None and value != "":
                    return value
            return None

        abstracts = [
            abstract for paper in group if (abstract := normalize_metadata_text(paper.abstract))
        ]
        abstract = max(abstracts, key=len) if abstracts else None
        title = normalize_metadata_text(base.title) or base.title
        normalized_dois = _unique(
            value for paper in [base, *group] if (value := normalize_doi(paper.doi))
        )
        publisher_dois = [doi for doi in normalized_dois if normalize_arxiv_id(doi) is None]
        doi = (
            publisher_dois[0]
            if publisher_dois
            else (normalized_dois[0] if normalized_dois else None)
        )
        landing_urls = _unique_urls(
            value
            for paper in [base, *group]
            for value in ([paper.url] if paper.url else []) + paper.alternate_urls
        )
        pdf_urls = _unique_urls(
            value
            for paper in [base, *group]
            for value in ([paper.pdf_url] if paper.pdf_url else []) + paper.alternate_pdf_urls
        )
        arxiv_ids = _unique(value for paper in [base, *group] for value in _arxiv_identities(paper))
        arxiv_id = arxiv_ids[0] if arxiv_ids else None
        citations = [paper.citation_count for paper in group if paper.citation_count is not None]
        authors = max((paper.authors for paper in group), key=len, default=[])
        sources = _unique(source for paper in group for source in paper.source_names)
        return Paper(
            id=stable_paper_id(doi=doi, arxiv_id=arxiv_id, title=title),
            title=title,
            abstract=abstract,
            authors=authors,
            publication_year=first("publication_year"),
            publication_date=first("publication_date"),
            venue=first("venue"),
            doi=doi,
            arxiv_id=arxiv_id,
            arxiv_ids=arxiv_ids,
            openalex_id=first("openalex_id"),
            semantic_scholar_id=first("semantic_scholar_id"),
            url=landing_urls[0] if landing_urls else None,
            alternate_urls=landing_urls[1:],
            pdf_url=pdf_urls[0] if pdf_urls else None,
            alternate_pdf_urls=pdf_urls[1:],
            citation_count=max(citations) if citations else None,
            source_names=sources,
        )
