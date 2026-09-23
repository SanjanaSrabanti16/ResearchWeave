from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import httpx
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

from app.models.document import (
    DocumentChunk,
    DocumentReference,
    DocumentSection,
    ParsedPaper,
    SourcePDF,
)
from app.parsers.base import PDFParserError, PDFParserUnavailableError

TEI = "{http://www.tei-c.org/ns/1.0}"
logger = logging.getLogger(__name__)

_PUBLISHER_BOILERPLATE = (
    "to cite this article",
    "view the article online",
    "downloaded from",
    "recent citations",
    "all rights reserved",
    "copyright",
)


class ParseQuality(StrEnum):
    HEALTHY = "healthy"
    FRONT_MATTER_DEGRADED = "front_matter_degraded"
    SEVERELY_DEGRADED = "severely_degraded"


@dataclass(frozen=True)
class FrontMatterHealth:
    quality: ParseQuality
    reasons: tuple[str, ...]


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = " ".join("".join(element.itertext()).split())
    return value or None


def _author(element: ET.Element) -> str | None:
    parts = [value for node in element.findall(f".//{TEI}persName/*") if (value := _text(node))]
    if not parts:
        return _text(element.find(f".//{TEI}persName"))
    return " ".join(parts)


def _chunk_id(pdf_sha256: str, section_id: str, index: int, text: str) -> str:
    value = f"{pdf_sha256}\0{section_id}\0{index}\0{text}".encode()
    return f"chunk-{hashlib.sha256(value).hexdigest()[:20]}"


def _readable_ratio(value: str) -> float:
    compact = [character for character in value if not character.isspace()]
    if not compact:
        return 0.0
    readable = sum(
        character.isalnum() or character in ".,;:!?()[]{}'-/–—%&+=" for character in compact
    )
    return readable / len(compact)


def _has_publisher_boilerplate(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return any(cue in normalized for cue in _PUBLISHER_BOILERPLATE)


def _trustworthy_title(value: str | None) -> bool:
    if value is None:
        return False
    normalized = " ".join(value.split())
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", normalized)
    return (
        8 <= len(normalized) <= 300
        and len(words) >= 2
        and _readable_ratio(normalized) >= 0.85
        and not _has_publisher_boilerplate(normalized)
    )


def _trustworthy_abstract(value: str | None) -> bool:
    if value is None:
        return False
    normalized = " ".join(value.split())
    return (
        len(normalized) >= 80
        and len(re.findall(r"[A-Za-z][A-Za-z'-]*", normalized)) >= 12
        and _readable_ratio(normalized) >= 0.85
        and not _has_publisher_boilerplate(normalized)
    )


def _body_text(document: ParsedPaper) -> str:
    return "\n".join(section.text for section in document.sections if section.text.strip())


def _body_is_healthy(document: ParsedPaper) -> bool:
    body = _body_text(document)
    return (
        len(body) >= 800
        and sum(bool(section.text.strip()) for section in document.sections) >= 2
        and _readable_ratio(body) >= 0.8
    )


def _first_numbered_heading(root: ET.Element) -> int | None:
    for heading in root.findall(f".//{TEI}text/{TEI}body//{TEI}head"):
        number = heading.attrib.get("n", "").strip()
        match = re.match(r"(\d+)", number)
        if match:
            return int(match.group(1))
    return None


def assess_front_matter(root: ET.Element, document: ParsedPaper) -> FrontMatterHealth:
    """Conservatively classify critical front matter without judging paper semantics."""
    reasons: list[str] = []
    if not _trustworthy_title(document.title):
        reasons.append("missing_or_degraded_title")
    if document.abstract and not _trustworthy_abstract(document.abstract):
        reasons.append("degraded_abstract")
    elif not document.abstract:
        reasons.append("missing_abstract")

    first_number = _first_numbered_heading(root)
    if first_number is not None and first_number > 1:
        reasons.append("body_section_sequence_starts_late")

    first_text = next((section.text for section in document.sections if section.text.strip()), "")
    if first_text:
        if _readable_ratio(first_text[:1000]) < 0.75:
            reasons.append("opening_text_is_malformed")
        elif re.match(r"^[a-z,.;:)\]]", first_text.lstrip()):
            reasons.append("opening_text_appears_truncated")

    if len(reasons) < 2:
        return FrontMatterHealth(ParseQuality.HEALTHY, tuple(reasons))
    if _body_is_healthy(document):
        return FrontMatterHealth(ParseQuality.FRONT_MATTER_DEGRADED, tuple(reasons))
    return FrontMatterHealth(ParseQuality.SEVERELY_DEGRADED, tuple(reasons))


class GrobidParser:
    name = "grobid"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        version: str,
        timeout_seconds: float,
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.timeout_seconds = timeout_seconds

    async def parse(self, paper_id: str, pdf_bytes: bytes, source_pdf: SourcePDF) -> ParsedPaper:
        try:
            response = await self.client.post(
                f"{self.base_url}/api/processFulltextDocument",
                files={"input": ("document.pdf", pdf_bytes, "application/pdf")},
                data={
                    "consolidateHeader": "0",
                    "consolidateCitations": "0",
                    "includeRawCitations": "1",
                },
                headers={"Accept": "application/xml"},
                timeout=self.timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise PDFParserUnavailableError("GROBID parser is unavailable") from exc
        if response.status_code == 503:
            raise PDFParserUnavailableError("GROBID parser is busy or unavailable")
        if response.status_code >= 400:
            raise PDFParserError(f"GROBID could not parse the PDF ({response.status_code})")
        try:
            root = ET.fromstring(response.content)
        except (ET.ParseError, DefusedXmlException) as exc:
            raise PDFParserError("GROBID returned invalid TEI XML") from exc
        document = self._to_document(paper_id, root, source_pdf)
        initial_health = assess_front_matter(root, document)
        fallback_fields: list[str] = []
        if initial_health.quality is not ParseQuality.HEALTHY:
            fallback_root = await self._header_fallback(pdf_bytes)
            if fallback_root is not None:
                fallback_fields = self._patch_front_matter(
                    document, fallback_root, source_pdf.sha256
                )

        final_health = assess_front_matter(root, document)
        provenance = self.name
        if fallback_fields:
            provenance += f"+grobid_header_fallback({','.join(fallback_fields)})"
        if final_health.quality is not ParseQuality.HEALTHY:
            provenance += f"[{final_health.quality.value}]"
            logger.warning(
                "PDF parse completed with %s front matter: %s",
                final_health.quality.value,
                ", ".join(final_health.reasons),
            )
        document.parser = provenance
        return document

    async def _header_fallback(self, pdf_bytes: bytes) -> ET.Element | None:
        """Retry only GROBID's bounded header extraction; never replace the body."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/processHeaderDocument",
                files={"input": ("document.pdf", pdf_bytes, "application/pdf")},
                data={
                    "startPage": "0",
                    "endPage": "2",
                    "consolidateHeader": "0",
                },
                headers={"Accept": "application/xml"},
                timeout=self.timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.NetworkError):
            logger.warning("GROBID header fallback was unavailable", exc_info=True)
            return None
        if response.status_code >= 400:
            logger.warning("GROBID header fallback failed with status %s", response.status_code)
            return None
        try:
            return ET.fromstring(response.content)
        except (ET.ParseError, DefusedXmlException):
            logger.warning("GROBID header fallback returned invalid TEI XML", exc_info=True)
            return None

    def _patch_front_matter(
        self,
        document: ParsedPaper,
        fallback_root: ET.Element,
        pdf_sha256: str,
    ) -> list[str]:
        header = fallback_root.find(f".//{TEI}teiHeader")
        if header is None:
            return []
        recovered: list[str] = []
        candidate_title = _text(header.find(f".//{TEI}titleStmt/{TEI}title"))
        if not _trustworthy_title(document.title) and _trustworthy_title(candidate_title):
            document.title = candidate_title
            recovered.append("title")

        abstract_node = header.find(f".//{TEI}profileDesc/{TEI}abstract")
        candidate_abstract = _text(abstract_node)
        if not _trustworthy_abstract(document.abstract) and _trustworthy_abstract(
            candidate_abstract
        ):
            document.abstract = candidate_abstract
            paragraphs = (
                [_text(node) for node in abstract_node.findall(f".//{TEI}p")]
                if abstract_node is not None
                else []
            )
            document.abstract_chunks = self._chunks(
                "abstract", paragraphs or [candidate_abstract], pdf_sha256
            )
            recovered.append("abstract")
        return recovered

    def _to_document(self, paper_id: str, root: ET.Element, source_pdf: SourcePDF) -> ParsedPaper:
        header = root.find(f".//{TEI}teiHeader")
        title = _text(header.find(f".//{TEI}titleStmt/{TEI}title") if header is not None else None)
        authors = self._authors(
            header.findall(f".//{TEI}sourceDesc//{TEI}author") if header is not None else []
        )
        abstract_node = (
            header.find(f".//{TEI}profileDesc/{TEI}abstract") if header is not None else None
        )
        abstract = _text(abstract_node)
        abstract_chunks = self._chunks(
            "abstract",
            [_text(node) for node in abstract_node.findall(f".//{TEI}p")]
            if abstract_node is not None
            else ([abstract] if abstract else []),
            source_pdf.sha256,
        )
        sections: list[DocumentSection] = []
        body = root.find(f".//{TEI}text/{TEI}body")
        if body is not None:
            self._sections(body, 1, source_pdf.sha256, sections)
        references = self._references(root.findall(f".//{TEI}listBibl/{TEI}biblStruct"))
        return ParsedPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            abstract_chunks=abstract_chunks,
            sections=sections,
            references=references,
            parser=self.name,
            parser_version=self.version,
            source_pdf=source_pdf,
        )

    @staticmethod
    def _authors(elements: Iterable[ET.Element]) -> list[str]:
        return [name for element in elements if (name := _author(element))]

    @staticmethod
    def _chunks(
        section_id: str, paragraphs: Iterable[str | None], pdf_sha256: str
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        offset = 0
        for value in paragraphs:
            if not value:
                continue
            index = len(chunks)
            chunks.append(
                DocumentChunk(
                    id=_chunk_id(pdf_sha256, section_id, index, value),
                    section_id=section_id,
                    text=value,
                    start_char=offset,
                    end_char=offset + len(value),
                )
            )
            offset += len(value) + 1
        return chunks

    def _sections(
        self,
        parent: ET.Element,
        level: int,
        pdf_sha256: str,
        sections: list[DocumentSection],
    ) -> None:
        for div in parent.findall(f"{TEI}div"):
            section_id = f"section-{len(sections) + 1:04d}"
            paragraphs = [_text(node) for node in div.findall(f"{TEI}p")]
            chunks = self._chunks(section_id, paragraphs, pdf_sha256)
            sections.append(
                DocumentSection(
                    id=section_id,
                    heading=_text(div.find(f"{TEI}head")),
                    level=level,
                    text="\n".join(chunk.text for chunk in chunks),
                    chunks=chunks,
                )
            )
            self._sections(div, level + 1, pdf_sha256, sections)

    def _references(self, elements: Iterable[ET.Element]) -> list[DocumentReference]:
        references: list[DocumentReference] = []
        for index, element in enumerate(elements, 1):
            date = element.find(f".//{TEI}date")
            when = date.attrib.get("when", "") if date is not None else ""
            match = re.search(r"\b(1[89]\d{2}|20\d{2}|2100)\b", when or (_text(date) or ""))
            identifiers = {
                node.attrib.get("type", "").casefold(): _text(node)
                for node in element.findall(f".//{TEI}idno")
            }
            references.append(
                DocumentReference(
                    id=f"reference-{index:04d}",
                    raw_text=_text(element.find(f".//{TEI}note[@type='raw_reference']"))
                    or _text(element),
                    title=_text(element.find(f".//{TEI}analytic/{TEI}title")),
                    authors=self._authors(element.findall(f".//{TEI}analytic/{TEI}author")),
                    year=int(match.group()) if match else None,
                    venue=_text(element.find(f".//{TEI}monogr/{TEI}title")),
                    doi=identifiers.get("doi"),
                    arxiv_id=identifiers.get("arxiv"),
                )
            )
        return references
