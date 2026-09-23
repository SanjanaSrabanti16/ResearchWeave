import hashlib

import httpx

from app.models.document import SourcePDF
from app.parsers.grobid import GrobidParser
from app.services.parsed_document_cache import ParsedDocumentCache

PDF = b"%PDF-1.7\nsynthetic front matter"
GOOD_TITLE = "A Trustworthy Study of Configurable Research Systems"
GOOD_ABSTRACT = (
    "This abstract describes a reproducible study of configurable research systems, "
    "including its design, evaluation procedure, observations, and bounded conclusions."
)
BODY_ONE = (
    "The study introduces a configurable workflow and evaluates it under controlled "
    "conditions. " * 20
)
BODY_TWO = (
    "The evaluation reports measured behavior, qualitative observations, and limitations. " * 20
)


def tei(
    *,
    title: str = GOOD_TITLE,
    abstract: str = GOOD_ABSTRACT,
    first_number: str = "1.",
    first_heading: str = "Introduction",
    first_text: str = BODY_ONE,
    second_text: str = BODY_TWO,
) -> str:
    abstract_xml = (
        f"<abstract><div><p>{abstract}</p></div></abstract>" if abstract else "<abstract/>"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>{title}</title></titleStmt>
      <sourceDesc><biblStruct><analytic/></biblStruct></sourceDesc>
    </fileDesc>
    <profileDesc>{abstract_xml}</profileDesc>
  </teiHeader>
  <text><body>
    <div><head n="{first_number}">{first_heading}</head><p>{first_text}</p></div>
    <div><head n="3.">Results</head><p>{second_text}</p></div>
  </body></text>
</TEI>"""


def header_tei(*, title: str = "", abstract: str = "") -> str:
    abstract_xml = (
        f"<abstract><div><p>{abstract}</p></div></abstract>" if abstract else "<abstract/>"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc><titleStmt><title>{title}</title></titleStmt></fileDesc>
    <profileDesc>{abstract_xml}</profileDesc>
  </teiHeader>
</TEI>"""


def source() -> SourcePDF:
    return SourcePDF(
        acquisition_method="upload",
        sha256=hashlib.sha256(PDF).hexdigest(),
        size_bytes=len(PDF),
    )


async def parse_with_responses(
    primary: str, fallback: str = "", version: str = "0.9.1-crf+frontmatter-v1"
):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/processFulltextDocument"):
            return httpx.Response(200, text=primary)
        if request.url.path.endswith("/processHeaderDocument"):
            return httpx.Response(200, text=fallback or header_tei())
        raise AssertionError(f"Unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        document = await GrobidParser(client, "http://grobid:8070", version, 5).parse(
            "paper-1", PDF, source()
        )
    return document, calls


async def test_healthy_grobid_parse_does_not_invoke_fallback() -> None:
    document, calls = await parse_with_responses(tei())

    assert calls == ["/api/processFulltextDocument"]
    assert document.parser == "grobid"
    assert document.title == GOOD_TITLE


async def test_missing_title_is_recovered_from_bounded_header_fallback() -> None:
    document, calls = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title=GOOD_TITLE),
    )

    assert calls == ["/api/processFulltextDocument", "/api/processHeaderDocument"]
    assert document.title == GOOD_TITLE


async def test_empty_abstract_is_recovered_only_from_explicit_abstract_node() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title=GOOD_TITLE, abstract=GOOD_ABSTRACT),
    )

    assert document.abstract == GOOD_ABSTRACT
    assert [chunk.text for chunk in document.abstract_chunks] == [GOOD_ABSTRACT]


async def test_missing_abstract_is_not_fabricated_without_recoverable_text() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title=GOOD_TITLE),
    )

    assert document.abstract is None
    assert document.abstract_chunks == []


async def test_publisher_cover_metadata_never_becomes_research_body() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title=GOOD_TITLE),
    )

    assert [section.heading for section in document.sections] == ["Introduction", "Results"]
    assert all("To cite this article" not in section.text for section in document.sections)


async def test_publisher_boilerplate_is_rejected_from_title_and_abstract() -> None:
    boilerplate = (
        "To cite this article, view the article online for updates and enhancements. "
        "This content was downloaded from a publisher website and all rights are reserved."
    )
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title="To cite this article", abstract=boilerplate),
    )

    assert document.title is None
    assert document.abstract is None


async def test_corrupted_front_matter_with_healthy_body_is_marked_degraded() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="2.", first_text="�" * 80 + "broken"),
        header_tei(),
    )

    assert "[front_matter_degraded]" in document.parser
    assert "[severely_degraded]" not in document.parser


async def test_healthy_later_grobid_sections_are_preserved_when_patched() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title=GOOD_TITLE),
    )

    assert [section.text for section in document.sections] == [BODY_ONE.strip(), BODY_TWO.strip()]
    assert len(document.sections) == 2


async def test_fallback_does_not_duplicate_existing_abstract_content() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract=GOOD_ABSTRACT, first_number="2."),
        header_tei(title=GOOD_TITLE, abstract=GOOD_ABSTRACT),
    )

    assert document.abstract == GOOD_ABSTRACT
    assert len(document.abstract_chunks) == 1
    assert document.parser.count("abstract") == 0


async def test_noisy_fallback_does_not_replace_healthy_grobid_title() -> None:
    document, _ = await parse_with_responses(
        tei(title=GOOD_TITLE, abstract="", first_number="2."),
        header_tei(title="���", abstract=GOOD_ABSTRACT),
    )

    assert document.title == GOOD_TITLE
    assert document.abstract == GOOD_ABSTRACT


async def test_missing_title_and_abstract_alone_are_not_severely_degraded() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="1."),
        header_tei(),
    )

    assert "[front_matter_degraded]" in document.parser
    assert "[severely_degraded]" not in document.parser


async def test_parser_provenance_identifies_fallback_derived_fields() -> None:
    document, _ = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title=GOOD_TITLE, abstract=GOOD_ABSTRACT),
    )

    assert "grobid_header_fallback(title,abstract)" in document.parser


def test_front_matter_parser_version_changes_cache_identity() -> None:
    old = ParsedDocumentCache.key(PDF, "grobid", "0.9.1-crf")
    new = ParsedDocumentCache.key(PDF, "grobid", "0.9.1-crf+frontmatter-v1")

    assert old != new


async def test_degraded_parse_calls_only_local_grobid_parser_endpoints() -> None:
    _, calls = await parse_with_responses(
        tei(title="", abstract="", first_number="2."),
        header_tei(title=GOOD_TITLE),
    )

    assert calls == ["/api/processFulltextDocument", "/api/processHeaderDocument"]
