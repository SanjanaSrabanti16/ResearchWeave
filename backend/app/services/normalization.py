from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
ARXIV_PREFIX = re.compile(
    r"^(?:https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/|/?(?:abs|pdf)/|arxiv:\s*)",
    re.IGNORECASE,
)
ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
ARXIV_IDENTIFIER = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})", re.IGNORECASE)
ARXIV_DOI_PREFIX = "10.48550/arxiv."


class _MetadataTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, _tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(" ")

    def handle_endtag(self, _tag: str) -> None:
        self.parts.append(" ")

    def handle_startendtag(self, _tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def unknown_decl(self, data: str) -> None:
        if data.startswith("CDATA["):
            self.parts.append(data[6:])


def normalize_metadata_text(value: Any) -> str | None:
    """Return display-safe plain text from provider HTML/XML/JATS-like metadata."""
    text = clean_text(value)
    if not text:
        return None
    parser = _MetadataTextParser()
    try:
        parser.feed(html.unescape(text))
        parser.close()
    except (AssertionError, ValueError):
        return clean_text(html.unescape(re.sub(r"<[^>]*>", " ", text)))
    normalized = clean_text("".join(parser.parts))
    return re.sub(r"\s+([,.;:!?])", r"\1", normalized) if normalized else None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def normalize_doi(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    normalized = text
    while True:
        stripped = DOI_PREFIX.sub("", normalized).strip()
        if stripped == normalized:
            break
        normalized = stripped
    normalized = normalized.casefold()
    return normalized or None


def normalize_arxiv_id(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    normalized_doi = normalize_doi(text)
    if normalized_doi and normalized_doi.startswith(ARXIV_DOI_PREFIX):
        identifier = ARXIV_VERSION.sub("", normalized_doi[len(ARXIV_DOI_PREFIX) :]).strip()
        return identifier if ARXIV_IDENTIFIER.fullmatch(identifier) else None
    parsed = urlparse(text)
    if parsed.hostname and parsed.hostname.casefold().rstrip(".") not in {
        "arxiv.org",
        "www.arxiv.org",
    }:
        return None
    normalized = ARXIV_PREFIX.sub("", text).split("?", 1)[0].split("#", 1)[0].strip(" /")
    normalized = re.sub(r"\.pdf$", "", normalized, flags=re.IGNORECASE)
    normalized = ARXIV_VERSION.sub("", normalized).strip().casefold()
    return normalized if ARXIV_IDENTIFIER.fullmatch(normalized) else None


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def stable_paper_id(*, doi: str | None, arxiv_id: str | None, title: str) -> str:
    identity = (
        f"doi:{doi}"
        if doi
        else f"arxiv:{arxiv_id}"
        if arxiv_id
        else f"title:{normalize_title(title)}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"^(\d{4})", value)
    return int(match.group(1)) if match else None
