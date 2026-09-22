from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

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
        return self._to_document(paper_id, root, source_pdf)

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
