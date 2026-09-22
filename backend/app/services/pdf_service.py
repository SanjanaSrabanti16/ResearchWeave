from __future__ import annotations

import hashlib
import re

from app.models.document import (
    ParsedPaper,
    PDFAcquisitionRequest,
    PDFProcessingResponse,
    SourcePDF,
)
from app.parsers.base import PDFParser
from app.providers.unpaywall import UnpaywallError, UnpaywallProvider
from app.services.parsed_document_cache import ParsedDocumentCache
from app.services.pdf_validation import PDFValidationError, SecurePDFDownloader, validate_pdf_bytes


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
    ) -> ParsedPaper:
        validate_pdf_bytes(pdf_bytes, self.max_bytes)
        cache_key = self.cache.key(pdf_bytes, self.parser.name, self.parser.version)
        source = SourcePDF(
            acquisition_method=acquisition_method,
            url=source_url,
            sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            size_bytes=len(pdf_bytes),
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"paper_id": paper_id, "source_pdf": source})
        document = await self.parser.parse(paper_id, pdf_bytes, source)
        self.cache.put(cache_key, document)
        return document


class PDFAcquisitionService:
    def __init__(
        self,
        downloader: SecurePDFDownloader,
        unpaywall: UnpaywallProvider,
        processor: PDFProcessingService,
    ) -> None:
        self.downloader = downloader
        self.unpaywall = unpaywall
        self.processor = processor

    @staticmethod
    def arxiv_pdf_url(arxiv_id: str) -> str:
        normalized = arxiv_id.strip()
        normalized = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", normalized)
        normalized = re.sub(r"^arxiv:", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"v\d+$", "", normalized).removesuffix(".pdf")
        pattern = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})"
        if not re.fullmatch(pattern, normalized, re.IGNORECASE):
            raise PDFValidationError("The arXiv identifier is invalid")
        return f"https://arxiv.org/pdf/{normalized}.pdf"

    async def acquire(self, request: PDFAcquisitionRequest) -> PDFProcessingResponse:
        candidates: list[tuple[str, str, bool]] = []
        if request.pdf_url:
            candidates.append(("existing_pdf", request.pdf_url, False))
        if request.arxiv_id:
            candidates.append(("arxiv", self.arxiv_pdf_url(request.arxiv_id), False))
        errors: list[str] = []
        for method, url, trusted_discovery in candidates:
            try:
                pdf_bytes, final_url = await self.downloader.download(
                    url, trusted_discovery=trusted_discovery
                )
                document = await self.processor.parse(
                    request.paper_id, pdf_bytes, method, final_url
                )
                return PDFProcessingResponse(
                    status=method,
                    paper_id=request.paper_id,
                    document=document,
                    message="Validated PDF acquired and parsed successfully.",
                )
            except PDFValidationError as exc:
                errors.append(str(exc))

        if request.doi:
            try:
                url = await self.unpaywall.find_pdf(request.doi)
            except UnpaywallError as exc:
                errors.append(str(exc))
            else:
                if url:
                    try:
                        pdf_bytes, final_url = await self.downloader.download(
                            url, trusted_discovery=True
                        )
                        document = await self.processor.parse(
                            request.paper_id, pdf_bytes, "unpaywall", final_url
                        )
                        return PDFProcessingResponse(
                            status="unpaywall",
                            paper_id=request.paper_id,
                            document=document,
                            message="Validated PDF acquired and parsed successfully.",
                        )
                    except PDFValidationError as exc:
                        errors.append(str(exc))

        message = "No validated open-access PDF was found. Upload a PDF you may lawfully use."
        if errors:
            message = f"{message} Last acquisition error: {errors[-1]}"
        return PDFProcessingResponse(
            status="upload_required", paper_id=request.paper_id, message=message
        )

    async def upload(self, paper_id: str, pdf_bytes: bytes) -> PDFProcessingResponse:
        document = await self.processor.parse(paper_id, pdf_bytes, "upload")
        return PDFProcessingResponse(
            status="upload",
            paper_id=paper_id,
            document=document,
            message="Uploaded PDF validated and parsed successfully.",
        )
