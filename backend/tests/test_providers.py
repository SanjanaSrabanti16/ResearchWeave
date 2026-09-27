from datetime import UTC, datetime

import arxiv

from app.providers.arxiv import ArxivProvider
from app.providers.openalex import OpenAlexProvider
from app.providers.semantic_scholar import SemanticScholarProvider


def test_openalex_normalization_reconstructs_abstract() -> None:
    paper = OpenAlexProvider.normalize(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Visual Agents",
            "publication_year": 2024,
            "publication_date": "2024-03-02",
            "ids": {"doi": "https://doi.org/10.2/ABC"},
            "authorships": [{"author": {"display_name": "Ada Author"}}],
            "abstract_inverted_index": {"Agents": [1], "Visual": [0]},
            "primary_location": {
                "landing_page_url": "https://example.test/paper",
                "source": {"display_name": "VIS"},
            },
            "best_oa_location": {"pdf_url": "https://example.test/paper.pdf"},
            "cited_by_count": 12,
        }
    )
    assert paper is not None
    assert paper.abstract == "Visual Agents"
    assert paper.doi == "10.2/abc"
    assert paper.authors == ["Ada Author"]


def test_openalex_normalization_preserves_multiple_oa_locations() -> None:
    paper = OpenAlexProvider.normalize(
        {
            "id": "https://openalex.org/W2",
            "display_name": "Multiple OA Copies",
            "primary_location": {
                "landing_page_url": "https://publisher.example/article",
                "pdf_url": "https://publisher.example/article.pdf",
                "source": {"display_name": "Journal"},
            },
            "best_oa_location": {
                "landing_page_url": "https://repository.example/item",
                "pdf_url": "https://repository.example/item.pdf",
            },
            "locations": [
                {
                    "landing_page_url": "https://repository.example/item",
                    "pdf_url": "https://repository.example/item.pdf",
                },
                {
                    "landing_page_url": "https://second.example/item",
                    "pdf_url": "https://second.example/item.pdf",
                },
            ],
        }
    )

    assert paper is not None
    assert paper.url == "https://publisher.example/article"
    assert paper.alternate_urls == [
        "https://repository.example/item",
        "https://second.example/item",
        "https://openalex.org/W2",
    ]
    assert paper.pdf_url == "https://repository.example/item.pdf"
    assert paper.alternate_pdf_urls == [
        "https://publisher.example/article.pdf",
        "https://second.example/item.pdf",
    ]


def test_semantic_scholar_normalization() -> None:
    paper = SemanticScholarProvider.normalize(
        {
            "paperId": "S2",
            "title": "Agentic Visualization",
            "abstract": "An abstract",
            "year": 2023,
            "authors": [{"authorId": "1", "name": "Sam Scholar"}],
            "externalIds": {"DOI": "10.3/TEST", "ArXiv": "2301.00123v1"},
            "openAccessPdf": {"url": "https://example.test/open.pdf"},
            "citationCount": 5,
        }
    )
    assert paper is not None
    assert paper.semantic_scholar_id == "S2"
    assert paper.arxiv_id == "2301.00123"
    assert paper.pdf_url == "https://example.test/open.pdf"


def test_arxiv_result_normalization() -> None:
    result = arxiv.Result(
        entry_id="http://arxiv.org/abs/2401.01234v2",
        published=datetime(2024, 1, 2, tzinfo=UTC),
        title=" A paper\n title ",
        summary=" Summary text. ",
        authors=[arxiv.Result.Author("Alex Example")],
        doi="10.4/ABC",
        journal_ref="Journal of Examples",
        links=[
            arxiv.Result.Link(
                href="https://arxiv.org/pdf/2401.01234",
                title="pdf",
                content_type="application/pdf",
            )
        ],
    )
    paper = ArxivProvider.normalize(result)
    assert paper is not None
    assert paper.title == "A paper title"
    assert paper.abstract == "Summary text."
    assert paper.authors == ["Alex Example"]
    assert paper.publication_year == 2024
    assert paper.publication_date == "2024-01-02"
    assert paper.arxiv_id == "2401.01234"
    assert paper.doi == "10.4/abc"
    assert paper.venue == "Journal of Examples"
    assert paper.url == "http://arxiv.org/abs/2401.01234v2"
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.01234"
    assert paper.source_names == ["arxiv"]
