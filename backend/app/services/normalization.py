from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
ARXIV_PREFIX = re.compile(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)", re.IGNORECASE)
ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def normalize_doi(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    normalized = DOI_PREFIX.sub("", text).strip().casefold()
    return normalized or None


def normalize_arxiv_id(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    normalized = ARXIV_PREFIX.sub("", text).removesuffix(".pdf")
    normalized = ARXIV_VERSION.sub("", normalized).strip().casefold()
    return normalized or None


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
