from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Paper(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(min_length=1)
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    publication_date: str | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = Field(default=None, ge=0)
    source_names: list[str] = Field(default_factory=list)
    semantic_score: float | None = None
    reranker_score: float | None = None

    @field_validator("authors", "source_names", mode="before")
    @classmethod
    def none_to_list(cls, value: Any) -> Any:
        return [] if value is None else value


class PaperCandidate(Paper):
    """Provider-normalized paper before deduplication and ranking."""
