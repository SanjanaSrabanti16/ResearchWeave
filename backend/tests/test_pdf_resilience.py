from __future__ import annotations

import httpx

from app.models.document import ParsedPaper, PDFAcquisitionRequest, SourcePDF
from app.providers.crossref import CrossrefLocation, CrossrefProvider
from app.providers.europe_pmc import EuropePMCProvider
from app.providers.unpaywall import UnpaywallLocation
from app.services.landing_page_resolver import LandingPageResolver
from app.services.pdf_service import PDFAcquisitionService
from app.services.pdf_validation import PDFDownloadError, PDFValidationError, SecurePDFDownloader

PDF = b"%PDF-1.7\nresilient"


class Processor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def parse(
        self,
        paper_id: str,
        pdf_bytes: bytes,
        method: str,
        source_url: str | None = None,
        acquisition_provenance: str | None = None,
    ) -> ParsedPaper:
        self.calls.append((method, source_url or ""))
        return ParsedPaper(
            paper_id=paper_id,
            parser="mock-grobid",
            parser_version="1",
            source_pdf=SourcePDF(
                acquisition_method=method,
                acquisition_provenance=acquisition_provenance,
                url=source_url,
                sha256="a" * 64,
                size_bytes=len(pdf_bytes),
            ),
        )


class Downloader:
    def __init__(self, outcomes: dict[str, bytes | Exception] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[str] = []

    async def download(self, url: str, *, trusted_discovery: bool = False):
        del trusted_discovery
        self.calls.append(url)
        outcome = self.outcomes.get(url, PDF)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, url


class LocationsUnpaywall:
    def __init__(self, locations: list[UnpaywallLocation] | None = None) -> None:
        self.locations = locations or []

    async def find_locations(self, _: str) -> list[UnpaywallLocation]:
        return self.locations


class NoCrossref:
    async def find_locations(self, _: str) -> list[CrossrefLocation]:
        return []


class FakePMC:
    def __init__(self, pmcid: str | None = None) -> None:
        self.pmcid = pmcid

    async def resolve_pmcids(self, _doi: str | None, _known: list[str]) -> list[str]:
        return [self.pmcid] if self.pmcid else []

    @staticmethod
    def pdf_locations(pmcid: str) -> list[str]:
        return [f"https://europepmc.org/articles/{pmcid}?pdf=render"]


class FakeLanding:
    def __init__(self, mapping: dict[str, tuple[list[str], bool]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    async def find_pdf_urls(self, url: str) -> tuple[list[str], str, bool]:
        self.calls.append(url)
        urls, paywall = self.mapping.get(url, ([], False))
        return urls, url, paywall


def service(
    downloader: Downloader,
    *,
    unpaywall: LocationsUnpaywall | None = None,
    crossref=None,
    pmc=None,
    landing=None,
    processor: Processor | None = None,
) -> PDFAcquisitionService:
    return PDFAcquisitionService(
        downloader,
        unpaywall or LocationsUnpaywall(),
        processor or Processor(),
        crossref=crossref,
        europe_pmc=pmc,
        landing_pages=landing,
    )


async def test_openalex_outage_does_not_block_doi_unpaywall_acquisition() -> None:
    url = "https://repository.example/open.pdf"
    result = await service(
        Downloader(),
        unpaywall=LocationsUnpaywall([UnpaywallLocation(url, True, "repository")]),
    ).acquire(PDFAcquisitionRequest(paper_id="p", doi="10.1000/openalex-discovered"))
    assert result.status == "unpaywall"
    assert result.attempts[-1].outcome == "success"


async def test_semantic_scholar_outage_does_not_block_known_arxiv_identity() -> None:
    downloader = Downloader()
    result = await service(downloader).acquire(
        PDFAcquisitionRequest(paper_id="p", arxiv_id="2501.01234")
    )
    assert result.status == "arxiv"
    assert downloader.calls == ["https://arxiv.org/pdf/2501.01234"]


async def test_cached_canonical_metadata_survives_all_search_provider_outages() -> None:
    downloader = Downloader()
    result = await service(downloader).acquire(
        PDFAcquisitionRequest(
            paper_id="p",
            doi="10.1000/cached",
            pdf_url="https://publisher.example/cached.pdf",
            alternate_urls=["https://publisher.example/article"],
        )
    )
    assert result.status == "existing_pdf"
    assert downloader.calls == ["https://publisher.example/cached.pdf"]


async def test_html_pdf_candidate_and_403_fall_through_to_later_pdf() -> None:
    html = "https://publisher.example/not-pdf.pdf"
    denied = "https://publisher.example/denied.pdf"
    valid = "https://repository.example/valid.pdf"
    downloader = Downloader(
        {
            html: PDFValidationError("The remote resource is not served as a PDF"),
            denied: PDFDownloadError("The PDF server rejected the request (403)"),
        }
    )
    result = await service(downloader).acquire(
        PDFAcquisitionRequest(paper_id="p", pdf_url=html, alternate_pdf_urls=[denied, valid])
    )
    assert downloader.calls == [html, denied, valid]
    assert [attempt.outcome for attempt in result.attempts] == [
        "invalid_pdf",
        "forbidden",
        "success",
    ]


async def test_known_id_extensionless_form_preempts_problematic_suffix() -> None:
    with_extension = "https://arxiv.org/pdf/2501.01234.pdf"
    extensionless = "https://arxiv.org/pdf/2501.01234"
    downloader = Downloader({with_extension: PDFDownloadError("rejected (406)")})
    result = await service(downloader).acquire(
        PDFAcquisitionRequest(
            paper_id="p",
            pdf_url=with_extension,
            alternate_pdf_urls=[extensionless],
            arxiv_id="2501.01234",
        )
    )
    assert downloader.calls == [extensionless]
    assert result.status == "arxiv"


async def test_unpaywall_best_failure_uses_alternate_oa_location() -> None:
    best = "https://publisher.example/best.pdf"
    alternate = "https://repository.example/alternate.pdf"
    downloader = Downloader({best: PDFDownloadError("rejected (403)")})
    result = await service(
        downloader,
        unpaywall=LocationsUnpaywall(
            [
                UnpaywallLocation(best, True, "publisher"),
                UnpaywallLocation(alternate, True, "repository"),
            ]
        ),
    ).acquire(PDFAcquisitionRequest(paper_id="p", doi="10.1000/alternate"))
    assert downloader.calls == [best, alternate]
    assert result.status == "unpaywall"


async def test_doi_and_mdpi_landing_pages_discover_public_pdf_metadata() -> None:
    doi_landing = "https://doi.org/10.3390/example"
    mdpi_landing = "https://www.mdpi.com/2079-9292/1/2/3"
    mdpi_pdf = "https://mdpi-res.com/article.pdf"
    landing = FakeLanding(
        {
            doi_landing: ([mdpi_pdf], False),
            mdpi_landing: ([mdpi_pdf], False),
        }
    )
    result = await service(Downloader(), crossref=NoCrossref(), landing=landing).acquire(
        PDFAcquisitionRequest(paper_id="p", doi="10.3390/example", url=mdpi_landing)
    )
    assert result.status == "publisher"
    assert result.document is not None
    assert result.document.source_pdf.url == mdpi_pdf
    assert any(attempt.outcome == "pdf_link_discovered" for attempt in result.attempts)


async def test_pmc_route_succeeds_when_identifier_is_available() -> None:
    downloader = Downloader()
    result = await service(downloader, pmc=FakePMC("PMC12345")).acquire(
        PDFAcquisitionRequest(paper_id="p", pmcid="PMC12345")
    )
    assert result.status == "pmc"
    assert downloader.calls == ["https://europepmc.org/articles/PMC12345?pdf=render"]


async def test_all_routes_fail_cleanly_and_paywall_is_never_parsed() -> None:
    landing_url = "https://publisher.example/paywall"
    processor = Processor()
    result = await service(
        Downloader(),
        landing=FakeLanding({landing_url: ([], True)}),
        processor=processor,
    ).acquire(PDFAcquisitionRequest(paper_id="p", url=landing_url))
    assert result.status == "upload_required"
    assert "authentication" in (result.message or "").casefold()
    assert processor.calls == []


async def test_failed_attempt_is_not_negative_cached_and_new_route_can_succeed() -> None:
    url = "https://repository.example/new.pdf"
    downloader = Downloader({url: PDFDownloadError("rejected (503)")})
    resolver = LocationsUnpaywall([UnpaywallLocation(url, True)])
    acquisition = service(downloader, unpaywall=resolver)
    first = await acquisition.acquire(PDFAcquisitionRequest(paper_id="p", doi="10.1/new"))
    downloader.outcomes[url] = PDF
    second = await acquisition.acquire(PDFAcquisitionRequest(paper_id="p", doi="10.1/new"))
    assert first.status == "upload_required"
    assert second.status == "unpaywall"


async def public_resolver(_: str) -> list[str]:
    return ["93.184.216.34"]


async def test_real_landing_parser_finds_relative_citation_pdf_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/article":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<meta name="citation_pdf_url" content="/article.pdf">',
            )
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = SecurePDFDownloader(
            client, max_bytes=1000, timeout_seconds=2, resolver=public_resolver
        )
        processor = Processor()
        result = await PDFAcquisitionService(
            downloader,
            LocationsUnpaywall(),
            processor,
            landing_pages=LandingPageResolver(downloader),
        ).acquire(PDFAcquisitionRequest(paper_id="p", url="https://www.mdpi.com/article"))
    assert result.status == "publisher"
    assert processor.calls == [("publisher", "https://www.mdpi.com/article.pdf")]


async def test_crossref_and_europe_pmc_public_metadata_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "crossref" in request.url.host:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "URL": "https://publisher.example/article",
                        "link": [
                            {
                                "URL": "https://publisher.example/article.pdf",
                                "content-type": "application/pdf",
                            }
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={"resultList": {"result": [{"pmcid": "PMC98765"}]}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        crossref = await CrossrefProvider(client).find_locations("10.1000/example")
        pmcids = await EuropePMCProvider(client).resolve_pmcids("10.1000/example", [])
    assert any(location.is_pdf for location in crossref)
    assert any(not location.is_pdf for location in crossref)
    assert pmcids == ["PMC98765"]
