from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourcePDF(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    acquisition_method: Literal["existing_pdf", "arxiv", "unpaywall", "upload"]
    url: str | None = None
    sha256: str
    size_bytes: int = Field(ge=1)


class DocumentChunk(BaseModel):
    id: str
    section_id: str
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class DocumentSection(BaseModel):
    id: str
    heading: str | None = None
    level: int = Field(ge=1)
    text: str
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    chunks: list[DocumentChunk] = Field(default_factory=list)


class DocumentReference(BaseModel):
    id: str
    raw_text: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None


class ParsedPaper(BaseModel):
    paper_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    abstract_chunks: list[DocumentChunk] = Field(default_factory=list)
    sections: list[DocumentSection] = Field(default_factory=list)
    references: list[DocumentReference] = Field(default_factory=list)
    parser: str
    parser_version: str
    source_pdf: SourcePDF


class PDFAcquisitionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    paper_id: str = Field(min_length=1, max_length=200)
    doi: str | None = Field(default=None, max_length=300)
    arxiv_id: str | None = Field(default=None, max_length=100)
    pdf_url: str | None = Field(default=None, max_length=2048)


class PDFProcessingResponse(BaseModel):
    status: Literal["existing_pdf", "arxiv", "unpaywall", "upload", "upload_required"]
    paper_id: str
    document: ParsedPaper | None = None
    message: str | None = None
