import httpx

from app.models.document import ParsedPaper, PDFAcquisitionRequest, SourcePDF
from app.models.paper import PaperCandidate
from app.providers.unpaywall import UnpaywallProvider
from app.services.pdf_service import PDFAcquisitionService
from app.services.pdf_validation import PDFDownloadError, PDFValidationError


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

    async def parse(
        self,
        paper_id: str,
        pdf_bytes: bytes,
        method: str,
        source_url: str | None = None,
        acquisition_provenance: str | None = None,
    ):
        return ParsedPaper(
            paper_id=paper_id,
            parser="mock",
            parser_version="1",
            source_pdf=SourcePDF(
                acquisition_method=method,
                acquisition_provenance=acquisition_provenance,
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
    assert result.acquisition_provenance == "arxiv_id"
    assert result.attempts[0].outcome == "success"


async def test_no_available_source_requests_upload() -> None:
    service = PDFAcquisitionService(RecordingDownloader(), NoopUnpaywall(), RecordingProcessor())
    result = await service.acquire(PDFAcquisitionRequest(paper_id="paper-1"))
    assert result.status == "upload_required"
    assert result.document is None


class SequenceDownloader:
    def __init__(self, outcomes: dict[str, bytes | Exception] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[tuple[str, bool]] = []

    async def download(self, url: str, *, trusted_discovery: bool = False):
        self.calls.append((url, trusted_discovery))
        outcome = self.outcomes.get(url, b"%PDF-1.7\ntest")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, url


class StaticUnpaywall:
    def __init__(self, url: str | None = None) -> None:
        self.url = url
        self.calls: list[str] = []

    async def find_pdf(self, doi: str) -> str | None:
        self.calls.append(doi)
        return self.url


class FakeArxivProvider:
    def __init__(self, papers: list[PaperCandidate] | None = None) -> None:
        self.papers = papers or []
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int, *_: object) -> list[PaperCandidate]:
        self.calls.append((query, limit))
        return self.papers


async def test_known_direct_pdf_succeeds_first_and_stops() -> None:
    downloader = SequenceDownloader()
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor())

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            pdf_url="https://arxiv.org/direct.pdf",
            arxiv_id="2401.12345",
        )
    )

    assert [call[0] for call in downloader.calls] == ["https://arxiv.org/direct.pdf"]
    assert result.status == "existing_pdf"
    assert result.acquisition_provenance == "known_pdf_url"
    assert result.attempts[0].outcome == "success"


async def test_failed_direct_pdf_falls_through_to_preserved_arxiv_id() -> None:
    direct = "https://arxiv.org/missing.pdf"
    arxiv_url = "https://arxiv.org/pdf/2401.12345.pdf"
    downloader = SequenceDownloader({direct: PDFDownloadError("rejected (404)")})
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor())

    result = await service.acquire(
        PDFAcquisitionRequest(paper_id="paper-1", pdf_url=direct, arxiv_id="2401.12345v2")
    )

    assert [call[0] for call in downloader.calls] == [direct, arxiv_url]
    assert [attempt.outcome for attempt in result.attempts] == ["not_found", "success"]
    assert result.acquisition_provenance == "arxiv_id"


async def test_failed_direct_uses_alternate_pdf_before_arxiv() -> None:
    direct = "https://arxiv.org/missing.pdf"
    alternate = "https://openreview.net/working.pdf"
    downloader = SequenceDownloader({direct: PDFDownloadError("rejected (404)")})
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor())

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            pdf_url=direct,
            alternate_pdf_urls=[alternate],
            arxiv_id="2401.12345",
        )
    )

    assert [call[0] for call in downloader.calls] == [direct, alternate]
    assert result.acquisition_provenance == "alternate_pdf_url"


async def test_arxiv_failure_falls_through_to_unpaywall_candidate() -> None:
    arxiv_url = "https://arxiv.org/pdf/2401.12345.pdf"
    oa_url = "https://repository.example/paper.pdf"
    downloader = SequenceDownloader({arxiv_url: PDFDownloadError("rejected (403)")})
    unpaywall = StaticUnpaywall(oa_url)
    service = PDFAcquisitionService(downloader, unpaywall, RecordingProcessor())

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1", arxiv_id="2401.12345", doi="https://doi.org/10.1/ABC"
        )
    )

    assert [call[0] for call in downloader.calls] == [arxiv_url, oa_url]
    assert downloader.calls[-1][1] is True
    assert unpaywall.calls == ["10.1/abc"]
    assert result.status == "unpaywall"
    assert result.acquisition_provenance == "unpaywall"


async def test_duplicate_pdf_urls_are_attempted_once() -> None:
    url = "https://arxiv.org/same.pdf"
    downloader = SequenceDownloader({url: PDFDownloadError("network failure")})
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor())

    await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            pdf_url=url,
            alternate_pdf_urls=[url, "https://arxiv.org/same.pdf/"],
        )
    )

    assert [call[0] for call in downloader.calls] == [url]


async def test_acquisition_attempt_count_is_bounded() -> None:
    urls = [f"https://arxiv.org/failure-{index}.pdf" for index in range(20)]
    downloader = SequenceDownloader({url: PDFDownloadError("network failure") for url in urls})
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor())

    result = await service.acquire(
        PDFAcquisitionRequest(paper_id="paper-1", pdf_url=urls[0], alternate_pdf_urls=urls[1:])
    )

    assert len(downloader.calls) == 12
    assert len(result.attempts) == 12


async def test_publisher_record_can_acquire_with_preserved_arxiv_identity() -> None:
    service = PDFAcquisitionService(SequenceDownloader(), StaticUnpaywall(), RecordingProcessor())
    request = PDFAcquisitionRequest(
        paper_id="doi:paper",
        doi="10.1/publisher",
        url="https://publisher.example/article",
        alternate_urls=["https://arxiv.org/abs/2401.54321v3"],
    )

    result = await service.acquire(request)

    assert result.status == "arxiv"
    assert result.document is not None
    assert result.document.source_pdf.url == "https://arxiv.org/pdf/2401.54321.pdf"


def fallback_paper(
    *,
    title: str = "A Robust Method: For Science",
    authors: list[str] | None = None,
    year: int = 2024,
    doi: str | None = None,
) -> PaperCandidate:
    return PaperCandidate(
        title=title,
        authors=authors or ["Ada Researcher"],
        publication_year=year,
        doi=doi,
        arxiv_id="2401.11111v2",
        source_names=["arxiv"],
    )


async def test_exact_normalized_title_fallback_succeeds_with_author_and_year() -> None:
    provider = FakeArxivProvider([fallback_paper()])
    service = PDFAcquisitionService(
        SequenceDownloader(), StaticUnpaywall(), RecordingProcessor(), provider
    )

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            title="a robust method—for science!",
            authors=["Ada Researcher"],
            publication_year=2024,
        )
    )

    assert provider.calls == [("a robust method—for science!", 5)]
    assert result.status == "arxiv"
    assert result.acquisition_provenance == "title_verified_arxiv_fallback"


async def test_near_but_different_arxiv_title_is_rejected() -> None:
    provider = FakeArxivProvider([fallback_paper(title="A Robust Method for Other Science")])
    downloader = SequenceDownloader()
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor(), provider)

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            title="A Robust Method for Science",
            authors=["Ada Researcher"],
            publication_year=2024,
        )
    )

    assert result.status == "upload_required"
    assert not downloader.calls
    assert result.attempts[-1].outcome == "title_match_rejected"


async def test_exact_title_with_incompatible_authors_is_rejected() -> None:
    provider = FakeArxivProvider([fallback_paper(authors=["Different Author"])])
    service = PDFAcquisitionService(
        SequenceDownloader(), StaticUnpaywall(), RecordingProcessor(), provider
    )

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            title="A Robust Method: For Science",
            authors=["Ada Researcher"],
            publication_year=2024,
        )
    )

    assert result.status == "upload_required"
    assert result.attempts[-1].outcome == "title_match_rejected"


async def test_exact_title_with_incompatible_year_is_rejected_without_doi() -> None:
    provider = FakeArxivProvider([fallback_paper(year=2018)])
    service = PDFAcquisitionService(
        SequenceDownloader(), StaticUnpaywall(), RecordingProcessor(), provider
    )

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            title="A Robust Method: For Science",
            authors=["Ada Researcher"],
            publication_year=2024,
        )
    )

    assert result.status == "upload_required"
    assert result.attempts[-1].outcome == "title_match_rejected"


async def test_compatible_doi_strongly_confirms_exact_title_fallback() -> None:
    provider = FakeArxivProvider(
        [fallback_paper(authors=["Different Author"], year=2020, doi="10.1/CONFIRMED")]
    )
    service = PDFAcquisitionService(
        SequenceDownloader(), StaticUnpaywall(), RecordingProcessor(), provider
    )

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1",
            title="A Robust Method for Science",
            authors=["Ada Researcher"],
            publication_year=2024,
            doi="https://doi.org/10.1/confirmed",
        )
    )

    assert result.acquisition_provenance == "title_verified_arxiv_fallback"


async def test_weak_title_similarity_alone_never_downloads() -> None:
    provider = FakeArxivProvider([fallback_paper(authors=["Ada Researcher"])])
    downloader = SequenceDownloader()
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor(), provider)

    result = await service.acquire(
        PDFAcquisitionRequest(
            paper_id="paper-1", title="Robust Science", authors=["Ada Researcher"]
        )
    )

    assert result.status == "upload_required"
    assert not downloader.calls


async def test_failed_candidates_retain_safe_diagnostics_without_secrets() -> None:
    secret_url = "https://arxiv.org/missing.pdf?api_key=do-not-log"
    downloader = SequenceDownloader({secret_url: PDFValidationError("not a valid PDF")})
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor())

    result = await service.acquire(PDFAcquisitionRequest(paper_id="paper-1", pdf_url=secret_url))
    rendered = result.model_dump_json()

    assert result.attempts[0].safe_identifier == "arxiv.org"
    assert result.attempts[0].outcome == "invalid_pdf"
    assert "do-not-log" not in rendered


async def test_previous_negative_result_does_not_suppress_new_route() -> None:
    downloader = SequenceDownloader()
    provider = FakeArxivProvider()
    service = PDFAcquisitionService(downloader, StaticUnpaywall(), RecordingProcessor(), provider)

    first = await service.acquire(PDFAcquisitionRequest(paper_id="paper-1"))
    second = await service.acquire(PDFAcquisitionRequest(paper_id="paper-1", arxiv_id="2401.12345"))

    assert first.status == "upload_required"
    assert second.status == "arxiv"


async def test_normal_single_pdf_url_behavior_remains_compatible() -> None:
    service = PDFAcquisitionService(SequenceDownloader(), StaticUnpaywall(), RecordingProcessor())

    result = await service.acquire(
        PDFAcquisitionRequest(paper_id="paper-1", pdf_url="https://arxiv.org/paper.pdf")
    )

    assert result.status == "existing_pdf"
    assert result.document is not None
    assert result.document.source_pdf.acquisition_method == "existing_pdf"


async def test_manual_upload_behavior_and_provenance_remain_available() -> None:
    service = PDFAcquisitionService(SequenceDownloader(), StaticUnpaywall(), RecordingProcessor())

    result = await service.upload("paper-1", b"%PDF-1.7\ntest")

    assert result.status == "upload"
    assert result.acquisition_provenance == "upload"
    assert result.document is not None
    assert result.document.source_pdf.acquisition_method == "upload"
    assert result.document.source_pdf.acquisition_provenance == "upload"
