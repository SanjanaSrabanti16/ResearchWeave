from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourcePDF(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    acquisition_method: Literal[
        "existing_pdf", "arxiv", "unpaywall", "crossref", "pmc", "publisher", "upload"
    ]
    acquisition_provenance: (
        Literal[
            "known_pdf_url",
            "alternate_pdf_url",
            "arxiv_id",
            "unpaywall",
            "crossref",
            "pmc",
            "publisher_landing_page",
            "doi_landing_page",
            "title_verified_arxiv_fallback",
            "upload",
        ]
        | None
    ) = None
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
    arxiv_ids: list[str] = Field(default_factory=list, max_length=20)
    pmcid: str | None = Field(default=None, max_length=40)
    pmcids: list[str] = Field(default_factory=list, max_length=20)
    pdf_url: str | None = Field(default=None, max_length=2048)
    alternate_pdf_urls: list[str] = Field(default_factory=list, max_length=20)
    url: str | None = Field(default=None, max_length=2048)
    alternate_urls: list[str] = Field(default_factory=list, max_length=20)
    title: str | None = Field(default=None, max_length=1000)
    authors: list[str] = Field(default_factory=list, max_length=100)
    publication_year: int | None = Field(default=None, ge=1800, le=2100)


class PDFAcquisitionAttempt(BaseModel):
    route: str
    candidate_type: str
    safe_identifier: str
    outcome: Literal[
        "success",
        "not_found",
        "forbidden",
        "invalid_pdf",
        "unsafe_url",
        "redirect_rejected",
        "timeout",
        "provider_no_oa",
        "title_match_rejected",
        "network_error",
        "pdf_link_discovered",
        "no_pdf_link",
        "authentication_required",
        "circuit_open",
        "deadline_exceeded",
    ]


class PDFProcessingResponse(BaseModel):
    status: Literal[
        "existing_pdf",
        "arxiv",
        "unpaywall",
        "crossref",
        "pmc",
        "publisher",
        "upload",
        "upload_required",
    ]
    paper_id: str
    document: ParsedPaper | None = None
    message: str | None = None
    acquisition_provenance: str | None = None
    attempts: list[PDFAcquisitionAttempt] = Field(default_factory=list)
