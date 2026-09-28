from __future__ import annotations

import hashlib
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
from app.providers.unpaywall import UnpaywallError, UnpaywallProvider
from app.services.normalization import normalize_arxiv_id, normalize_doi, normalize_title
from app.services.parsed_document_cache import ParsedDocumentCache
from app.services.pdf_validation import PDFValidationError, SecurePDFDownloader, validate_pdf_bytes

MAX_PDF_CANDIDATES = 12


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
    ) -> None:
        self.downloader = downloader
        self.unpaywall = unpaywall
        self.processor = processor
        self.arxiv_provider = arxiv_provider

    @staticmethod
    def arxiv_pdf_url(arxiv_id: str) -> str:
        normalized = normalize_arxiv_id(arxiv_id)
        pattern = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})"
        if not normalized or not re.fullmatch(pattern, normalized, re.IGNORECASE):
            raise PDFValidationError("The arXiv identifier is invalid")
        return f"https://arxiv.org/pdf/{normalized}.pdf"

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
        try:
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
        except PDFValidationError as exc:
            attempts.append(self._attempt(candidate, self._outcome(exc)))
            return None
        attempts.append(self._attempt(candidate, "success"))
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
        if request.pdf_url:
            candidates.append(
                _PDFCandidate(
                    "known_pdf_url",
                    "direct_pdf_url",
                    request.pdf_url,
                    "existing_pdf",
                    "existing_pdf",
                    "known_pdf_url",
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
        normalized_ids: list[str] = []
        for value in explicit_arxiv_ids:
            if not value:
                continue
            normalized = normalize_arxiv_id(value)
            if normalized and normalized not in normalized_ids:
                normalized_ids.append(normalized)
            elif normalized is None:
                attempts.append(
                    PDFAcquisitionAttempt(
                        route="arxiv_id",
                        candidate_type="arxiv_id",
                        safe_identifier="invalid-arxiv-id",
                        outcome="unsafe_url",
                    )
                )
        identity_values = [
            request.doi,
            request.url,
            *request.alternate_urls,
            request.pdf_url,
            *request.alternate_pdf_urls,
        ]
        for value in identity_values:
            normalized = normalize_arxiv_id(value)
            if normalized and normalized not in normalized_ids:
                normalized_ids.append(normalized)
        for arxiv_id in normalized_ids:
            candidate = _PDFCandidate(
                "arxiv_id",
                "arxiv_id",
                self.arxiv_pdf_url(arxiv_id),
                "arxiv",
                "arxiv",
                "arxiv_id",
            )
            candidates.append(candidate)
        candidates.extend(
            _PDFCandidate(
                "alternate_pdf_url",
                "provider_pdf_url",
                url,
                "existing_pdf",
                "existing_pdf",
                "alternate_pdf_url",
            )
            for url in request.alternate_pdf_urls
            if url not in arxiv_pdf_urls
        )
        return self._deduplicate_candidates(candidates, attempts)

    @classmethod
    def _deduplicate_candidates(
        cls,
        candidates: list[_PDFCandidate],
        attempts: list[PDFAcquisitionAttempt],
        attempted_urls: set[str] | None = None,
    ) -> list[_PDFCandidate]:
        seen = attempted_urls if attempted_urls is not None else set()
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

    async def acquire(self, request: PDFAcquisitionRequest) -> PDFProcessingResponse:
        attempts: list[PDFAcquisitionAttempt] = []
        attempted_urls: set[str] = set()
        candidates = self._initial_candidates(request, attempts)
        for candidate in candidates:
            attempted_urls.add(self._url_key(candidate.url))
            result = await self._try_candidate(request, candidate, attempts)
            if result is not None:
                return result

        doi = normalize_doi(request.doi)
        if doi and len(attempts) < MAX_PDF_CANDIDATES:
            try:
                url = await self.unpaywall.find_pdf(doi)
            except UnpaywallError as exc:
                message = str(exc).casefold()
                outcome = (
                    "forbidden"
                    if "(401)" in message or "(403)" in message
                    else "not_found"
                    if "(404)" in message
                    else "network_error"
                )
                attempts.append(
                    PDFAcquisitionAttempt(
                        route="unpaywall",
                        candidate_type="doi_oa_discovery",
                        safe_identifier="api.unpaywall.org",
                        outcome=outcome,
                    )
                )
            else:
                if url:
                    candidate = _PDFCandidate(
                        "unpaywall",
                        "doi_oa_pdf_url",
                        url,
                        "unpaywall",
                        "unpaywall",
                        "unpaywall",
                        True,
                    )
                    if self._url_key(url) not in attempted_urls:
                        attempted_urls.add(self._url_key(url))
                        result = await self._try_candidate(request, candidate, attempts)
                        if result is not None:
                            return result
                else:
                    attempts.append(
                        PDFAcquisitionAttempt(
                            route="unpaywall",
                            candidate_type="doi_oa_discovery",
                            safe_identifier="api.unpaywall.org",
                            outcome="provider_no_oa",
                        )
                    )

        if self.arxiv_provider is not None and request.title and len(attempts) < MAX_PDF_CANDIDATES:
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
                    candidate = _PDFCandidate(
                        "title_verified_arxiv_fallback",
                        "verified_arxiv_title_match",
                        self.arxiv_pdf_url(match_id),
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

        message = "No validated open-access PDF was found. Upload a PDF you may lawfully use."
        return PDFProcessingResponse(
            status="upload_required",
            paper_id=request.paper_id,
            message=message,
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
