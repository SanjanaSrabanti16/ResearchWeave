import httpx

from app.models.document import ParsedPaper, PDFAcquisitionRequest, SourcePDF
from app.providers.unpaywall import UnpaywallProvider
from app.services.pdf_service import PDFAcquisitionService


async def test_unpaywall_uses_best_open_access_pdf_and_identifying_email() -> None:
    captured: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        captured = request
        return httpx.Response(
            200,
            json={
                "is_oa": True,
                "best_oa_location": {"url_for_pdf": "https://repository.example/paper.pdf"},
                "oa_locations": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await UnpaywallProvider(client, "researcher@example.org").find_pdf(
            "10.1000/example"
        )

    assert result == "https://repository.example/paper.pdf"
    assert captured is not None
    assert captured.url.params["email"] == "researcher@example.org"
    assert "10.1000%2Fexample" in str(captured.url)


async def test_unpaywall_is_skipped_without_email() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as client:
        assert await UnpaywallProvider(client, None).find_pdf("10.1000/example") is None


async def test_unpaywall_malformed_response_is_rejected() -> None:
    from app.providers.unpaywall import UnpaywallError

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not-json"))
    ) as client:
        provider = UnpaywallProvider(client, "researcher@example.org")
        try:
            await provider.find_pdf("10.1000/example")
        except UnpaywallError as exc:
            assert str(exc) == "Unpaywall returned an invalid response"
        else:
            raise AssertionError("Malformed Unpaywall JSON should fail")


class RecordingDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def download(self, url: str, *, trusted_discovery: bool = False):
        self.calls.append((url, trusted_discovery))
        return b"%PDF-1.7\ntest", url


class NoopUnpaywall:
    async def find_pdf(self, _: str) -> None:
        return None


class RecordingProcessor:
    max_bytes = 100

    async def parse(self, paper_id: str, pdf_bytes: bytes, method: str, source_url: str):
        return ParsedPaper(
            paper_id=paper_id,
            parser="mock",
            parser_version="1",
            source_pdf=SourcePDF(
                acquisition_method=method,
                url=source_url,
                sha256="a" * 64,
                size_bytes=len(pdf_bytes),
            ),
        )


async def test_arxiv_acquisition_constructs_and_validates_official_pdf_url() -> None:
    downloader = RecordingDownloader()
    service = PDFAcquisitionService(downloader, NoopUnpaywall(), RecordingProcessor())

    result = await service.acquire(
        PDFAcquisitionRequest(paper_id="paper-1", arxiv_id="2401.12345v2")
    )

    assert result.status == "arxiv"
    assert downloader.calls == [("https://arxiv.org/pdf/2401.12345.pdf", False)]
    assert result.document is not None
    assert result.document.source_pdf.acquisition_method == "arxiv"


async def test_no_available_source_requests_upload() -> None:
    service = PDFAcquisitionService(RecordingDownloader(), NoopUnpaywall(), RecordingProcessor())
    result = await service.acquire(PDFAcquisitionRequest(paper_id="paper-1"))
    assert result.status == "upload_required"
    assert result.document is None
