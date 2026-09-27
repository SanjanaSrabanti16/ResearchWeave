import asyncio
import tempfile
from pathlib import Path

import httpx
import pytest
from starlette.datastructures import Headers, UploadFile

from app.models.document import SourcePDF
from app.parsers import PDFParserUnavailableError
from app.parsers.grobid import GrobidParser
from app.services.parsed_document_cache import ParsedDocumentCache
from app.services.pdf_service import PDFProcessingService
from app.services.pdf_upload import read_pdf_upload
from app.services.pdf_validation import (
    PDFDownloadError,
    PDFTooLargeError,
    PDFValidationError,
    SecurePDFDownloader,
    validate_pdf_bytes,
    validate_remote_url,
)

PDF = b"%PDF-1.7\nsmall test"
TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Structured Research Agents</title></titleStmt>
      <sourceDesc><biblStruct><analytic>
        <author><persName><forename>Sanjana</forename><surname>Researcher</surname></persName></author>
      </analytic></biblStruct></sourceDesc>
    </fileDesc>
    <profileDesc><abstract><div><p>An evidence-ready abstract.</p></div></abstract></profileDesc>
  </teiHeader>
  <text>
    <body><div><head>Introduction</head><p>First paragraph.</p>
      <div><head>Method</head><p>Second paragraph.</p></div>
    </div></body>
    <back><div><listBibl><biblStruct>
      <analytic><title>Prior Work</title>
        <author><persName><surname>Scholar</surname></persName></author>
      </analytic>
      <monogr><title>Journal</title><imprint><date when="2024"/></imprint></monogr>
      <idno type="DOI">10.1000/example</idno>
      <note type="raw_reference">Scholar, Prior Work, 2024.</note>
    </biblStruct></listBibl></div></back>
  </text>
</TEI>"""


async def public_resolver(_: str) -> list[str]:
    return ["93.184.216.34"]


def test_pdf_magic_and_size_validation() -> None:
    validate_pdf_bytes(PDF, 100)
    with pytest.raises(PDFValidationError, match="not a valid PDF"):
        validate_pdf_bytes(b"<html>not a pdf</html>", 100)
    with pytest.raises(PDFTooLargeError, match="exceeds"):
        validate_pdf_bytes(PDF, 5)


async def test_upload_temporary_file_is_closed_after_reading() -> None:
    temporary = tempfile.SpooledTemporaryFile(max_size=1)
    temporary.write(PDF)
    temporary.seek(0)
    upload = UploadFile(
        file=temporary,
        filename="paper.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )

    assert await read_pdf_upload(upload, 100) == PDF
    assert temporary.closed


async def test_secure_download_validates_redirects_type_magic_and_size() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://arxiv.org/paper.pdf"})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = SecurePDFDownloader(
            client, max_bytes=100, timeout_seconds=2, resolver=public_resolver
        )
        content, final_url = await downloader.download("https://arxiv.org/start")

    assert content == PDF
    assert final_url == "https://arxiv.org/paper.pdf"
    assert calls == ["https://arxiv.org/start", "https://arxiv.org/paper.pdf"]


async def test_remote_url_rejects_private_network_targets() -> None:
    async def private_resolver(_: str) -> list[str]:
        return ["127.0.0.1"]

    with pytest.raises(PDFDownloadError, match="private or local"):
        await validate_remote_url(
            "https://arxiv.org/paper.pdf",
            resolver=private_resolver,
            allowed_hosts=("arxiv.org",),
        )


async def test_remote_url_rejects_localhost_and_http() -> None:
    async def local_resolver(_: str) -> list[str]:
        return ["127.0.0.1"]

    with pytest.raises(PDFDownloadError, match="private or local"):
        await validate_remote_url("https://localhost/paper.pdf", resolver=local_resolver)
    with pytest.raises(PDFDownloadError, match="public HTTPS"):
        await validate_remote_url("http://arxiv.org/paper.pdf", resolver=public_resolver)


async def test_secure_download_rejects_redirect_to_private_host() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://localhost/private.pdf"})

    async def resolver(host: str) -> list[str]:
        return ["127.0.0.1"] if host == "localhost" else ["93.184.216.34"]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = SecurePDFDownloader(
            client, max_bytes=100, timeout_seconds=2, resolver=resolver
        )
        with pytest.raises(PDFDownloadError, match="approved scholarly|private or local"):
            await downloader.download("https://arxiv.org/start")


@pytest.mark.parametrize(
    ("headers", "content", "message"),
    [
        ({"content-type": "text/html"}, b"<html>login</html>", "not served as a PDF"),
        ({"content-type": "application/pdf"}, b"not-a-pdf", "not a valid PDF"),
        (
            {"content-type": "application/pdf", "content-length": "101"},
            PDF,
            "exceeds",
        ),
    ],
)
async def test_secure_download_rejects_non_pdf_and_oversized_content(
    headers: dict[str, str], content: bytes, message: str
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers=headers, content=content)
        )
    ) as client:
        downloader = SecurePDFDownloader(
            client, max_bytes=100, timeout_seconds=2, resolver=public_resolver
        )
        with pytest.raises(PDFValidationError, match=message):
            await downloader.download("https://arxiv.org/paper.pdf")


async def test_grobid_tei_mapping_has_stable_evidence_chunks() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "application/xml"}, text=TEI)
    )
    source = SourcePDF(
        acquisition_method="upload",
        sha256="a" * 64,
        size_bytes=len(PDF),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        parser = GrobidParser(client, "http://grobid:8070", "0.9.0-crf", 5)
        first = await parser.parse("paper-1", PDF, source)
        second = await parser.parse("paper-1", PDF, source)

    assert first.title == "Structured Research Agents"
    assert first.authors == ["Sanjana Researcher"]
    assert first.abstract == "An evidence-ready abstract."
    assert [section.heading for section in first.sections] == ["Introduction", "Method"]
    assert [section.level for section in first.sections] == [1, 2]
    assert first.sections[0].chunks[0].text == "First paragraph."
    assert first.sections[0].chunks[0].id == second.sections[0].chunks[0].id
    assert first.references[0].doi == "10.1000/example"
    assert first.references[0].year == 2024
    assert first.references[0].venue == "Journal"


async def test_grobid_unavailability_is_reported_without_affecting_search() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    source = SourcePDF(
        acquisition_method="upload",
        sha256="a" * 64,
        size_bytes=len(PDF),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
        parser = GrobidParser(client, "http://grobid:8070", "0.9.0-crf", 5)
        with pytest.raises(PDFParserUnavailableError, match="unavailable"):
            await parser.parse("paper-1", PDF, source)


class CountingParser:
    name = "test-parser"
    version = "1"

    def __init__(self) -> None:
        self.calls = 0

    async def parse(self, paper_id: str, _: bytes, source: SourcePDF):
        from app.models.document import ParsedPaper

        self.calls += 1
        return ParsedPaper(
            paper_id=paper_id,
            parser=self.name,
            parser_version=self.version,
            source_pdf=source,
        )


async def test_ident_parsed_document_cache_avoids_reparsing(tmp_path: Path) -> None:
    parser = CountingParser()
    service = PDFProcessingService(parser, ParsedDocumentCache(tmp_path), max_bytes=100)

    first = await service.parse("paper-1", PDF, "upload")
    second = await service.parse("paper-2", PDF, "upload")

    assert parser.calls == 1
    assert first.paper_id == "paper-1"
    assert second.paper_id == "paper-2"
    cached_files = await asyncio.to_thread(lambda: list(tmp_path.glob("*.json")))
    assert len(cached_files) == 1
