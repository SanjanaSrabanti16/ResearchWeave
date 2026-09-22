from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from app.models.paper import Paper, PaperCandidate
from app.services.normalization import (
    normalize_arxiv_id,
    normalize_doi,
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
            if arxiv_id := normalize_arxiv_id(paper.arxiv_id):
                if arxiv_id in arxiv_indexes:
                    union(index, arxiv_indexes[arxiv_id])
                else:
                    arxiv_indexes[arxiv_id] = index

        title_indexes: dict[str, list[int]] = defaultdict(list)
        for index, paper in enumerate(papers):
            if title := normalize_title(paper.title):
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
                if (doi := normalize_doi(paper.doi))
            }
            arxiv_ids = {
                arxiv_id
                for root in roots
                for paper in members_by_root[root]
                if (arxiv_id := normalize_arxiv_id(paper.arxiv_id))
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
        def completeness(paper: PaperCandidate) -> tuple[int, int, int]:
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
            return populated, len(paper.abstract or ""), len(paper.authors)

        base = max(group, key=completeness)

        def first(attribute: str) -> object | None:
            for paper in [base, *group]:
                value = getattr(paper, attribute)
                if value is not None and value != "":
                    return value
            return None

        abstracts = [paper.abstract for paper in group if paper.abstract]
        abstract = max(abstracts, key=len) if abstracts else None
        doi = next((value for paper in group if (value := normalize_doi(paper.doi))), None)
        arxiv_id = next(
            (value for paper in group if (value := normalize_arxiv_id(paper.arxiv_id))), None
        )
        citations = [paper.citation_count for paper in group if paper.citation_count is not None]
        authors = max((paper.authors for paper in group), key=len, default=[])
        sources = _unique(source for paper in group for source in paper.source_names)
        return Paper(
            id=stable_paper_id(doi=doi, arxiv_id=arxiv_id, title=base.title),
            title=base.title,
            abstract=abstract,
            authors=authors,
            publication_year=first("publication_year"),
            publication_date=first("publication_date"),
            venue=first("venue"),
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=first("openalex_id"),
            semantic_scholar_id=first("semantic_scholar_id"),
            url=first("url"),
            pdf_url=first("pdf_url"),
            citation_count=max(citations) if citations else None,
            source_names=sources,
        )
