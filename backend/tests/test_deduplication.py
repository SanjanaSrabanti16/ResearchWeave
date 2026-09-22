from app.models.paper import PaperCandidate
from app.services.deduplication_service import DeduplicationService


def paper(title: str, source: str, **kwargs: object) -> PaperCandidate:
    return PaperCandidate(title=title, source_names=[source], **kwargs)


def test_same_doi_merges_and_preserves_metadata() -> None:
    papers = [
        paper("A GREAT PAPER", "openalex", doi="https://doi.org/10.1/ABC", citation_count=4),
        paper(
            "A Great Paper",
            "semantic_scholar",
            doi="10.1/abc",
            abstract="A longer and more useful abstract.",
            citation_count=9,
            semantic_scholar_id="s2-id",
        ),
    ]
    result = DeduplicationService().deduplicate(papers)
    assert len(result) == 1
    assert result[0].doi == "10.1/abc"
    assert result[0].citation_count == 9
    assert result[0].semantic_scholar_id == "s2-id"
    assert set(result[0].source_names) == {"openalex", "semantic_scholar"}


def test_same_arxiv_id_merges_even_with_versions() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("First title", "arxiv", arxiv_id="2401.01234v2"),
            paper("Published title", "semantic_scholar", arxiv_id="ARXIV:2401.01234"),
        ]
    )
    assert len(result) == 1
    assert result[0].arxiv_id == "2401.01234"


def test_exact_normalized_title_merges() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("AI Agents: For Visualization", "openalex"),
            paper("ai agents for visualization", "semantic_scholar"),
        ]
    )
    assert len(result) == 1


def test_similar_but_different_titles_do_not_merge() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("AI Agents for Visualization Systems", "openalex"),
            paper("AI Agents for Visualization System Design", "semantic_scholar"),
        ]
    )
    assert len(result) == 2


def test_same_title_with_conflicting_dois_does_not_merge() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("Identical title", "openalex", doi="10.1/a"),
            paper("Identical title", "semantic_scholar", doi="10.1/b"),
        ]
    )
    assert len(result) == 2


def test_same_title_with_conflicting_arxiv_ids_does_not_merge() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("Identical title", "arxiv", arxiv_id="2401.00001"),
            paper("Identical title", "semantic_scholar", arxiv_id="2401.00002"),
        ]
    )
    assert len(result) == 2


def test_identifier_free_record_does_not_bridge_conflicting_doi_groups() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("Identical title", "openalex", doi="10.1/a"),
            paper("Identical title", "arxiv"),
            paper("Identical title", "semantic_scholar", doi="10.1/b"),
        ]
    )
    assert len(result) == 3


def test_same_title_with_one_missing_identifier_merges_safely() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("Identical title", "openalex", doi="10.1/a"),
            paper("Identical title", "semantic_scholar", abstract="More metadata"),
        ]
    )
    assert len(result) == 1
    assert result[0].doi == "10.1/a"


def test_same_doi_merges_even_when_titles_differ() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper("Preprint title", "openalex", doi="10.1/shared"),
            paper("Published title", "semantic_scholar", doi="10.1/shared"),
        ]
    )
    assert len(result) == 1


def test_same_arxiv_id_merges_even_when_titles_differ_and_dois_conflict() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper(
                "Preprint title",
                "arxiv",
                arxiv_id="2401.01234",
                doi="10.1/preprint",
            ),
            paper(
                "Published title",
                "semantic_scholar",
                arxiv_id="2401.01234v2",
                doi="10.1/published",
            ),
        ]
    )
    assert len(result) == 1
