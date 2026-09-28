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


def test_canonical_merge_sanitizes_title_and_abstract_metadata() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper(
                "A <italic>Great</italic> Paper",
                "semantic_scholar",
                abstract="A <bold>useful</bold> result with <tex-math>x^2</tex-math>.",
            ),
            paper("A Great Paper", "openalex", abstract="Short result."),
        ]
    )

    assert len(result) == 1
    assert result[0].title == "A Great Paper"
    assert result[0].abstract == "A useful result with x^2."


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


def test_publisher_canonical_record_preserves_arxiv_acquisition_identity() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper(
                "Published Paper Title",
                "openalex",
                doi="https://doi.org/10.1000/PAPER",
                publication_year=2025,
                venue="Journal of Reliable Results",
                url="https://publisher.example/article",
            ),
            paper(
                "Published Paper Title",
                "arxiv",
                arxiv_id="arXiv:2501.01234v2",
                venue="arXiv",
                url="https://arxiv.org/abs/2501.01234v2",
                pdf_url="https://arxiv.org/pdf/2501.01234v2.pdf",
                abstract="A much longer preprint abstract that must not make arXiv canonical.",
            ),
        ]
    )

    assert len(result) == 1
    merged = result[0]
    assert merged.title == "Published Paper Title"
    assert merged.venue == "Journal of Reliable Results"
    assert merged.publication_year == 2025
    assert merged.doi == "10.1000/paper"
    assert merged.arxiv_id == "2501.01234"
    assert merged.arxiv_ids == ["2501.01234"]
    assert set(merged.source_names) == {"openalex", "arxiv"}
    assert merged.url == "https://publisher.example/article"
    assert merged.alternate_urls == ["https://arxiv.org/abs/2501.01234v2"]
    assert merged.pdf_url == "https://arxiv.org/pdf/2501.01234v2.pdf"


def test_merge_preserves_and_deduplicates_alternate_urls() -> None:
    result = DeduplicationService().deduplicate(
        [
            paper(
                "Same Work",
                "openalex",
                doi="10.1/same",
                url="https://publisher.example/work/",
                alternate_urls=["https://repository.example/item"],
                pdf_url="https://repository.example/work.pdf",
                alternate_pdf_urls=["https://mirror.example/work.pdf"],
            ),
            paper(
                "Same Work",
                "semantic_scholar",
                doi="https://doi.org/10.1/SAME",
                url="https://publisher.example/work",
                alternate_urls=["https://repository.example/item"],
                pdf_url="https://repository.example/work.pdf",
                alternate_pdf_urls=["https://mirror.example/work.pdf"],
            ),
        ]
    )[0]

    assert result.url == "https://publisher.example/work/"
    assert result.alternate_urls == ["https://repository.example/item"]
    assert result.pdf_url == "https://repository.example/work.pdf"
    assert result.alternate_pdf_urls == ["https://mirror.example/work.pdf"]


def test_merge_preserves_all_normalized_arxiv_ids_for_same_doi() -> None:
    merged = DeduplicationService().deduplicate(
        [
            paper("Shared DOI", "first", doi="10.1/shared", arxiv_id="2401.00001v1"),
            paper("Shared DOI", "second", doi="10.1/shared", arxiv_id="2402.00002v2"),
        ]
    )[0]

    assert merged.arxiv_id == "2401.00001"
    assert merged.arxiv_ids == ["2401.00001", "2402.00002"]


def test_poorer_late_duplicate_does_not_replace_canonical_metadata() -> None:
    merged = DeduplicationService().deduplicate(
        [
            paper(
                "Complete Publisher Title",
                "openalex",
                doi="10.1/canonical",
                venue="Journal",
                publication_year=2024,
                authors=["Ada Author"],
            ),
            paper("Incomplete title", "arxiv", doi="10.1/canonical", arxiv_id="2401.12345"),
        ]
    )[0]

    assert merged.title == "Complete Publisher Title"
    assert merged.venue == "Journal"
    assert merged.publication_year == 2024
    assert merged.authors == ["Ada Author"]


def test_streetweave_multi_provider_representations_merge_into_one_work() -> None:
    title = (
        "StreetWeave: A Declarative Grammar for Street-Overlaid Visualization of Multivariate Data"
    )
    result = DeduplicationService().deduplicate(
        [
            paper(
                title,
                "openalex",
                doi="10.1109/TVCG.2025.3634647",
                openalex_id="W4417003572",
                venue="IEEE Transactions on Visualization and Computer Graphics",
                url="https://doi.org/10.1109/TVCG.2025.3634647",
            ),
            paper(
                title,
                "openalex",
                doi="10.48550/arXiv.2508.07496",
                openalex_id="W4416242310",
                url="http://arxiv.org/abs/2508.07496",
                pdf_url="https://arxiv.org/pdf/2508.07496",
            ),
            paper(
                title,
                "semantic_scholar",
                doi="10.1109/tvcg.2025.3634647",
                arxiv_id="2508.07496",
                semantic_scholar_id="s2-streetweave",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13340627/",
            ),
        ]
    )

    assert len(result) == 1
    merged = result[0]
    assert merged.doi == "10.1109/tvcg.2025.3634647"
    assert merged.arxiv_id == "2508.07496"
    assert merged.openalex_id == "W4417003572"
    assert merged.semantic_scholar_id == "s2-streetweave"
    assert set(merged.source_names) == {"openalex", "semantic_scholar"}
    assert "https://arxiv.org/pdf/2508.07496" in [
        merged.pdf_url,
        *merged.alternate_pdf_urls,
    ]
    assert "https://pmc.ncbi.nlm.nih.gov/articles/PMC13340627/" in [
        merged.pdf_url,
        *merged.alternate_pdf_urls,
    ]
