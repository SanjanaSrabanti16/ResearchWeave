from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from app.models.document import (
    ParsedPaper,
    PDFAcquisitionAttempt,
    PDFAcquisitionRequest,
    PDFProcessingResponse,
    SourcePDF,
)
from app.parsers.base import PDFParser
from app.providers.crossref import CrossrefError, CrossrefProvider
from app.providers.europe_pmc import EuropePMCError, EuropePMCProvider
from app.providers.unpaywall import UnpaywallError, UnpaywallProvider
from app.services.landing_page_resolver import LandingPageResolver
from app.services.normalization import normalize_arxiv_id, normalize_doi, normalize_title
from app.services.parsed_document_cache import ParsedDocumentCache
from app.services.pdf_validation import PDFValidationError, SecurePDFDownloader, validate_pdf_bytes
from app.services.provider_reliability import CircuitBreaker, CircuitOpenError

MAX_PDF_CANDIDATES = 12
logger = logging.getLogger(__name__)


class ArxivSearchProvider(Protocol):
    def search(
        self,
        query: str,
        limit: int,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> Awaitable[list]: ...


@dataclass(frozen=True)
class _PDFCandidate:
    route: str
    candidate_type: str
    url: str
    status: str
    acquisition_method: str
    provenance: str
    trusted_discovery: bool = False
    kind: str = "pdf"


class PDFProcessingService:
    def __init__(self, parser: PDFParser, cache: ParsedDocumentCache, max_bytes: int) -> None:
        self.parser = parser
        self.cache = cache
        self.max_bytes = max_bytes

    async def parse(
        self,
        paper_id: str,
        pdf_bytes: bytes,
        acquisition_method: str,
        source_url: str | None = None,
        acquisition_provenance: str | None = None,
    ) -> ParsedPaper:
        validate_pdf_bytes(pdf_bytes, self.max_bytes)
        cache_key = self.cache.key(pdf_bytes, self.parser.name, self.parser.version)
        source = SourcePDF(
            acquisition_method=acquisition_method,
            acquisition_provenance=acquisition_provenance,
            url=source_url,
            sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            size_bytes=len(pdf_bytes),
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            document = cached.model_copy(update={"paper_id": paper_id, "source_pdf": source})
            self.cache.put_paper_state(document)
            return document
        document = await self.parser.parse(paper_id, pdf_bytes, source)
        self.cache.put(cache_key, document)
        self.cache.put_paper_state(document)
        return document


class PDFAcquisitionService:
    def __init__(
        self,
        downloader: SecurePDFDownloader,
        unpaywall: UnpaywallProvider,
        processor: PDFProcessingService,
        arxiv_provider: ArxivSearchProvider | None = None,
        crossref: CrossrefProvider | None = None,
        europe_pmc: EuropePMCProvider | None = None,
        landing_pages: LandingPageResolver | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        total_deadline_seconds: float = 180.0,
    ) -> None:
        self.downloader = downloader
        self.unpaywall = unpaywall
        self.processor = processor
        self.arxiv_provider = arxiv_provider
        self.crossref = crossref
        self.europe_pmc = europe_pmc
        self.landing_pages = landing_pages
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.total_deadline_seconds = total_deadline_seconds

    @staticmethod
    def arxiv_pdf_url(arxiv_id: str) -> str:
        normalized = normalize_arxiv_id(arxiv_id)
        pattern = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})"
        if not normalized or not re.fullmatch(pattern, normalized, re.IGNORECASE):
            raise PDFValidationError("The arXiv identifier is invalid")
        return f"https://arxiv.org/pdf/{normalized}.pdf"

    @classmethod
    def arxiv_pdf_urls(cls, arxiv_id: str) -> tuple[str, str]:
        with_extension = cls.arxiv_pdf_url(arxiv_id)
        return with_extension.removesuffix(".pdf"), with_extension

    @staticmethod
    def _known_arxiv_ids(request: PDFAcquisitionRequest) -> list[str]:
        values = [
            request.arxiv_id,
            *request.arxiv_ids,
            request.doi,
            request.url,
            *request.alternate_urls,
            request.pdf_url,
            *request.alternate_pdf_urls,
        ]
        result: list[str] = []
        for value in values:
            normalized = normalize_arxiv_id(value)
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    @staticmethod
    def _url_key(url: str) -> str:
        parsed = urlsplit(url.strip())
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        try:
            port = parsed.port
        except ValueError:
            return url.strip().casefold()
        netloc = hostname if port in {None, 443} else f"{hostname}:{port}"
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme.casefold(), netloc, path, parsed.query, ""))

    @staticmethod
    def _safe_identifier(url: str) -> str:
        return (urlsplit(url).hostname or "unparseable-url").casefold().rstrip(".")

    @staticmethod
    def _outcome(error: PDFValidationError) -> str:
        message = str(error).casefold()
        if (
            "private or local" in message
            or "approved scholarly" in message
            or "public https" in message
        ):
            return "unsafe_url"
        if "redirect" in message:
            return "redirect_rejected"
        if "(404)" in message:
            return "not_found"
        if "(401)" in message or "(403)" in message:
            return "forbidden"
        if "timed out" in message or "timeout" in message:
            return "timeout"
        if "valid pdf" in message or "served as a pdf" in message or "exceeds" in message:
            return "invalid_pdf"
        return "network_error"

    @staticmethod
    def _author_overlap(left: list[str], right: list[str]) -> bool:
        def identities(names: list[str]) -> set[str]:
            result: set[str] = set()
            for name in names:
                normalized = normalize_title(name)
                if not normalized:
                    continue
                result.add(normalized)
                surname = normalized.split()[-1]
                if len(surname) >= 4:
                    result.add(f"surname:{surname}")
            return result

        return bool(identities(left) & identities(right))

    @classmethod
    def _verified_arxiv_match(cls, request: PDFAcquisitionRequest, papers: list) -> object | None:
        requested_title = normalize_title(request.title)
        if not requested_title:
            return None
        requested_doi = normalize_doi(request.doi)
        for paper in papers:
            if normalize_title(getattr(paper, "title", None)) != requested_title:
                continue
            candidate_doi = normalize_doi(getattr(paper, "doi", None))
            doi_match = bool(requested_doi and candidate_doi and requested_doi == candidate_doi)
            candidate_year = getattr(paper, "publication_year", None)
            year_compatible = not (
                request.publication_year is not None
                and candidate_year is not None
                and abs(request.publication_year - candidate_year) > 1
            )
            if doi_match or (
                year_compatible
                and cls._author_overlap(request.authors, list(getattr(paper, "authors", [])))
            ):
                return paper
        return None

    @staticmethod
    def _attempt(
        candidate: _PDFCandidate, outcome: str, safe_identifier: str | None = None
    ) -> PDFAcquisitionAttempt:
        return PDFAcquisitionAttempt(
            route=candidate.route,
            candidate_type=candidate.candidate_type,
            safe_identifier=safe_identifier
            or PDFAcquisitionService._safe_identifier(candidate.url),
            outcome=outcome,
        )

    async def _try_candidate(
        self,
        request: PDFAcquisitionRequest,
        candidate: _PDFCandidate,
        attempts: list[PDFAcquisitionAttempt],
    ) -> PDFProcessingResponse | None:
        is_arxiv_pdf = (urlsplit(candidate.url).hostname or "").casefold().rstrip(".") in {
            "arxiv.org",
            "www.arxiv.org",
        } and urlsplit(candidate.url).path.casefold().startswith("/pdf/")
        operation_key = ("arxiv", "pdf")
        try:
            if is_arxiv_pdf:
                self.circuit_breaker.before_call(operation_key)
            pdf_bytes, final_url = await self.downloader.download(
                candidate.url, trusted_discovery=candidate.trusted_discovery
            )
            document = await self.processor.parse(
                request.paper_id,
                pdf_bytes,
                candidate.acquisition_method,
                final_url,
                candidate.provenance,
            )
        except CircuitOpenError:
            attempts.append(self._attempt(candidate, "circuit_open"))
            logger.info(
                "pdf_candidate_skipped route=%s host=%s outcome=circuit_open",
                candidate.route,
                self._safe_identifier(candidate.url),
            )
            return None
        except PDFValidationError as exc:
            outcome = self._outcome(exc)
            attempts.append(self._attempt(candidate, outcome))
            if is_arxiv_pdf and outcome in {"network_error", "timeout"}:
                self.circuit_breaker.record_failure(operation_key, outcome)
            logger.info(
                "pdf_candidate_completed route=%s host=%s outcome=%s",
                candidate.route,
                self._safe_identifier(candidate.url),
                outcome,
            )
            return None
        if is_arxiv_pdf:
            self.circuit_breaker.record_success(operation_key)
        attempts.append(self._attempt(candidate, "success"))
        logger.info(
            "pdf_candidate_completed route=%s host=%s outcome=success",
            candidate.route,
            self._safe_identifier(final_url),
        )
        return PDFProcessingResponse(
            status=candidate.status,
            paper_id=request.paper_id,
            document=document,
            message="Validated PDF acquired and parsed successfully.",
            acquisition_provenance=candidate.provenance,
            attempts=attempts,
        )

    def _initial_candidates(
        self, request: PDFAcquisitionRequest, attempts: list[PDFAcquisitionAttempt]
    ) -> list[_PDFCandidate]:
        candidates: list[_PDFCandidate] = []
        normalized_ids = self._known_arxiv_ids(request)
        for arxiv_id in normalized_ids:
            candidates.extend(
                _PDFCandidate(
                    "arxiv_id",
                    "arxiv_id",
                    url,
                    "arxiv",
                    "arxiv",
                    "arxiv_id",
                )
                for url in self.arxiv_pdf_urls(arxiv_id)
            )
        if request.pdf_url:
            candidates.append(
                _PDFCandidate(
                    "known_pdf_url",
                    "direct_pdf_url",
                    request.pdf_url,
                    "existing_pdf",
                    "existing_pdf",
                    "known_pdf_url",
                    True,
                )
            )
        arxiv_pdf_urls = [
            url
            for url in request.alternate_pdf_urls
            if normalize_arxiv_id(url)
            and (urlsplit(url).hostname or "").casefold().rstrip(".")
            in {"arxiv.org", "www.arxiv.org"}
            and urlsplit(url).path.casefold().startswith("/pdf/")
        ]
        candidates.extend(
            _PDFCandidate(
                "alternate_pdf_url",
                "provider_arxiv_pdf_url",
                url,
                "arxiv",
                "arxiv",
                "alternate_pdf_url",
            )
            for url in arxiv_pdf_urls
        )

        explicit_arxiv_ids = [request.arxiv_id, *request.arxiv_ids]
        for value in explicit_arxiv_ids:
            if not value:
                continue
            normalized = normalize_arxiv_id(value)
            if normalized is None:
                attempts.append(
                    PDFAcquisitionAttempt(
                        route="arxiv_id",
                        candidate_type="arxiv_id",
                        safe_identifier="invalid-arxiv-id",
                        outcome="unsafe_url",
                    )
                )
        candidates.extend(
            _PDFCandidate(
                "alternate_pdf_url",
                "provider_pdf_url",
                url,
                "existing_pdf",
                "existing_pdf",
                "alternate_pdf_url",
                True,
            )
            for url in request.alternate_pdf_urls
            if url not in arxiv_pdf_urls
        )
        candidates.extend(
            _PDFCandidate(
                "provider_landing_page",
                "landing_page",
                url,
                "publisher",
                "publisher",
                "publisher_landing_page",
                True,
                "landing",
            )
            for url in ([request.url] if request.url else []) + request.alternate_urls
        )
        return self._deduplicate_candidates(candidates, attempts)

    @classmethod
    def _deduplicate_candidates(
        cls,
        candidates: list[_PDFCandidate],
        attempts: list[PDFAcquisitionAttempt],
        attempted_urls: set[str] | None = None,
    ) -> list[_PDFCandidate]:
        seen = set(attempted_urls or ())
        result: list[_PDFCandidate] = []
        for candidate in candidates:
            key = cls._url_key(candidate.url)
            if key in seen:
                continue
            seen.add(key)
            if len(result) + len(attempts) >= MAX_PDF_CANDIDATES:
                break
            result.append(candidate)
        return result

    @staticmethod
    def _resolver_attempt(
        attempts: list[PDFAcquisitionAttempt], route: str, host: str, outcome: str
    ) -> None:
        attempts.append(
            PDFAcquisitionAttempt(
                route=route,
                candidate_type="metadata_resolver",
                safe_identifier=host,
                outcome=outcome,
            )
        )

    async def _metadata_candidates(
        self,
        request: PDFAcquisitionRequest,
        attempts: list[PDFAcquisitionAttempt],
    ) -> list[_PDFCandidate]:
        doi = normalize_doi(request.doi)
        candidates: list[_PDFCandidate] = []
        known_pmc_values = [
            value
            for value in [
                request.pmcid,
                *request.pmcids,
                request.url,
                *request.alternate_urls,
                request.pdf_url,
                *request.alternate_pdf_urls,
            ]
            if value
        ]
        if self.europe_pmc is not None:
            try:
                pmcids = await self.europe_pmc.resolve_pmcids(doi, known_pmc_values)
            except EuropePMCError:
                self._resolver_attempt(attempts, "europe_pmc", "europepmc.org", "network_error")
            else:
                for pmcid in pmcids:
                    candidates.extend(
                        _PDFCandidate(
                            "pmc",
                            "pmc_pdf_url",
                            url,
                            "pmc",
                            "pmc",
                            "pmc",
                            True,
                        )
                        for url in self.europe_pmc.pdf_locations(pmcid)
                    )
        if doi and hasattr(self.unpaywall, "find_locations"):
            try:
                locations = await self.unpaywall.find_locations(doi)
            except UnpaywallError:
                self._resolver_attempt(attempts, "unpaywall", "api.unpaywall.org", "network_error")
            else:
                for location in locations:
                    candidates.append(
                        _PDFCandidate(
                            "unpaywall",
                            "doi_oa_pdf_url" if location.is_pdf else "oa_landing_page",
                            location.url,
                            "unpaywall" if location.is_pdf else "publisher",
                            "unpaywall" if location.is_pdf else "publisher",
                            "unpaywall" if location.is_pdf else "publisher_landing_page",
                            True,
                            "pdf" if location.is_pdf else "landing",
                        )
                    )
        elif doi:
            try:
                url = await self.unpaywall.find_pdf(doi)
            except UnpaywallError:
                self._resolver_attempt(attempts, "unpaywall", "api.unpaywall.org", "network_error")
            else:
                if url:
                    candidates.append(
                        _PDFCandidate(
                            "unpaywall",
                            "doi_oa_pdf_url",
                            url,
                            "unpaywall",
                            "unpaywall",
                            "unpaywall",
                            True,
                        )
                    )
        if doi and self.crossref is not None:
            try:
                locations = await self.crossref.find_locations(doi)
            except CrossrefError:
                self._resolver_attempt(attempts, "crossref", "api.crossref.org", "network_error")
            else:
                for location in locations:
                    candidates.append(
                        _PDFCandidate(
                            "crossref",
                            "crossref_pdf_url" if location.is_pdf else "crossref_landing_page",
                            location.url,
                            "crossref" if location.is_pdf else "publisher",
                            "crossref" if location.is_pdf else "publisher",
                            "crossref" if location.is_pdf else "publisher_landing_page",
                            True,
                            "pdf" if location.is_pdf else "landing",
                        )
                    )
        if doi:
            candidates.append(
                _PDFCandidate(
                    "doi_landing_page",
                    "doi_landing_page",
                    f"https://doi.org/{doi}",
                    "publisher",
                    "publisher",
                    "doi_landing_page",
                    True,
                    "landing",
                )
            )
        return candidates

    async def _resolve_landing(
        self,
        candidate: _PDFCandidate,
        attempts: list[PDFAcquisitionAttempt],
    ) -> list[_PDFCandidate]:
        if self.landing_pages is None:
            attempts.append(self._attempt(candidate, "no_pdf_link"))
            return []
        try:
            urls, final_url, paywall = await self.landing_pages.find_pdf_urls(candidate.url)
        except PDFValidationError as exc:
            attempts.append(self._attempt(candidate, self._outcome(exc)))
            return []
        if not urls:
            attempts.append(
                self._attempt(
                    candidate,
                    "authentication_required" if paywall else "no_pdf_link",
                    self._safe_identifier(final_url),
                )
            )
            return []
        attempts.append(
            self._attempt(candidate, "pdf_link_discovered", self._safe_identifier(final_url))
        )
        return [
            _PDFCandidate(
                candidate.route,
                "landing_page_pdf_url",
                url,
                candidate.status,
                candidate.acquisition_method,
                candidate.provenance,
                True,
            )
            for url in urls
        ]

    @staticmethod
    def _failure_message(attempts: list[PDFAcquisitionAttempt]) -> str:
        outcomes = {attempt.outcome for attempt in attempts}
        if "authentication_required" in outcomes:
            return (
                "A publisher page was found, but the PDF appears to require authentication. "
                "Upload a PDF you may lawfully use."
            )
        if outcomes & {"network_error", "timeout"}:
            return (
                "Available PDF locations temporarily failed. Upload a PDF you may lawfully use "
                "or try again later."
            )
        if "no_pdf_link" in outcomes:
            return (
                "Publisher pages were found, but no public PDF link was available. "
                "Upload a PDF you may lawfully use."
            )
        return (
            "Open-access PDF could not be resolved from available sources. "
            "Upload a PDF you may lawfully use."
        )

    async def _acquire(
        self,
        request: PDFAcquisitionRequest,
        attempts: list[PDFAcquisitionAttempt],
    ) -> PDFProcessingResponse:
        attempted_urls: set[str] = set()
        known_arxiv_ids = self._known_arxiv_ids(request)
        logger.info(
            "pdf_acquisition_started paper_id=%s arxiv_id_known=%s direct_candidates=%s "
            "search_circuit=%s pdf_circuit=%s",
            request.paper_id,
            bool(known_arxiv_ids),
            len(known_arxiv_ids) * 2,
            self.circuit_breaker.state(("arxiv", "search")),
            self.circuit_breaker.state(("arxiv", "pdf")),
        )
        initial = self._initial_candidates(request, attempts)
        metadata = await self._metadata_candidates(request, attempts)
        candidates = self._deduplicate_candidates(
            [
                *(candidate for candidate in initial if candidate.kind == "pdf"),
                *(candidate for candidate in metadata if candidate.kind == "pdf"),
                *(candidate for candidate in initial if candidate.kind == "landing"),
                *(candidate for candidate in metadata if candidate.kind == "landing"),
            ],
            attempts,
        )
        index = 0
        while index < len(candidates) and len(attempts) < MAX_PDF_CANDIDATES:
            candidate = candidates[index]
            index += 1
            key = self._url_key(candidate.url)
            if key in attempted_urls:
                continue
            attempted_urls.add(key)
            if candidate.kind == "landing":
                discovered = await self._resolve_landing(candidate, attempts)
                candidates[index:index] = self._deduplicate_candidates(
                    discovered, attempts, attempted_urls
                )
                continue
            result = await self._try_candidate(request, candidate, attempts)
            if result is not None:
                return result

        if (
            not known_arxiv_ids
            and self.arxiv_provider is not None
            and request.title
            and len(attempts) < MAX_PDF_CANDIDATES
        ):
            try:
                papers = await self.arxiv_provider.search(request.title, 5)
            except Exception:
                attempts.append(
                    PDFAcquisitionAttempt(
                        route="title_verified_arxiv_fallback",
                        candidate_type="arxiv_title_search",
                        safe_identifier="export.arxiv.org",
                        outcome="network_error",
                    )
                )
            else:
                match = self._verified_arxiv_match(request, papers)
                match_id = normalize_arxiv_id(getattr(match, "arxiv_id", None)) if match else None
                if match_id:
                    for url in self.arxiv_pdf_urls(match_id):
                        candidate = _PDFCandidate(
                            "title_verified_arxiv_fallback",
                            "verified_arxiv_title_match",
                            url,
                            "arxiv",
                            "arxiv",
                            "title_verified_arxiv_fallback",
                        )
                        if self._url_key(candidate.url) not in attempted_urls:
                            result = await self._try_candidate(request, candidate, attempts)
                            if result is not None:
                                return result
                else:
                    attempts.append(
                        PDFAcquisitionAttempt(
                            route="title_verified_arxiv_fallback",
                            candidate_type="arxiv_title_search",
                            safe_identifier="export.arxiv.org",
                            outcome="title_match_rejected",
                        )
                    )

        return PDFProcessingResponse(
            status="upload_required",
            paper_id=request.paper_id,
            message=self._failure_message(attempts),
            attempts=attempts,
        )

    async def acquire(self, request: PDFAcquisitionRequest) -> PDFProcessingResponse:
        attempts: list[PDFAcquisitionAttempt] = []
        try:
            async with asyncio.timeout(self.total_deadline_seconds):
                return await self._acquire(request, attempts)
        except TimeoutError:
            attempts.append(
                PDFAcquisitionAttempt(
                    route="overall_deadline",
                    candidate_type="operation_deadline",
                    safe_identifier="pdf-acquisition",
                    outcome="deadline_exceeded",
                )
            )
            return PDFProcessingResponse(
                status="upload_required",
                paper_id=request.paper_id,
                message="PDF acquisition timed out. Upload a PDF you may lawfully use.",
                attempts=attempts,
            )

    async def upload(self, paper_id: str, pdf_bytes: bytes) -> PDFProcessingResponse:
        document = await self.processor.parse(
            paper_id, pdf_bytes, "upload", acquisition_provenance="upload"
        )
        return PDFProcessingResponse(
            status="upload",
            paper_id=paper_id,
            document=document,
            message="Uploaded PDF validated and parsed successfully.",
            acquisition_provenance="upload",
        )
