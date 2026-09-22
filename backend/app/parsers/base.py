from __future__ import annotations

from typing import Protocol

from app.models.document import ParsedPaper, SourcePDF


class PDFParserError(RuntimeError):
    """A safe error for expected parsing failures."""


class PDFParserUnavailableError(PDFParserError):
    """Raised when the configured parser service cannot be reached."""


class PDFParser(Protocol):
    name: str
    version: str

    async def parse(
        self, paper_id: str, pdf_bytes: bytes, source_pdf: SourcePDF
    ) -> ParsedPaper: ...
