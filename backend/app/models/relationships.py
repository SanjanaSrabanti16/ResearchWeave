from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RelationshipType(StrEnum):
    SHARED_PROBLEM = "shared_problem"
    SHARED_METHOD = "shared_method"
    SHARED_FINDING = "shared_finding"
    COMPLEMENTARY_CONTRIBUTION = "complementary_contribution"
    CONTRASTING_RESULT = "contrasting_result"
    SHARED_LIMITATION = "shared_limitation"
    SHARED_FUTURE_WORK = "shared_future_work"
    RELATED_APPLICATION = "related_application"
    OTHER = "other"


class ReviewDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"


def _reject_html(value: str) -> str:
    if "<" in value or ">" in value:
        raise ValueError("HTML markup is not allowed")
    return value


class RelationshipProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_paper_id: str = Field(min_length=1, max_length=200)
    target_paper_id: str = Field(min_length=1, max_length=200)
    requested_relationship_types: list[RelationshipType] | None = Field(
        default=None, min_length=1, max_length=len(RelationshipType)
    )


class GeneratedRelationshipItem(BaseModel):
    """Provider-facing structured output. Confidence is computed after validation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    relationship_type: RelationshipType
    summary: str = Field(min_length=3, max_length=800)
    source_evidence_ids: list[str] = Field(min_length=1, max_length=8)
    target_evidence_ids: list[str] = Field(min_length=1, max_length=8)

    _safe_summary = field_validator("summary")(_reject_html)


class GeneratedRelationshipResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationships: list[GeneratedRelationshipItem] = Field(min_length=1, max_length=30)


class RelationshipItem(GeneratedRelationshipItem):
    confidence: float = Field(ge=0, le=1)


class AnalysisProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_fingerprint: str = Field(min_length=64, max_length=64)
    provider: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=200)
    extraction_version: str = Field(min_length=1, max_length=100)


class RelationshipDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_index: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=100)


class RelationshipEvidence(BaseModel):
    """Compact provenance copied from an already-validated M2 insight reference."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    evidence_id: str
    insight_field: str
    claim: str
    quote: str
    section_id: str | None = None
    section_heading: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class RelationshipProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    source_paper_id: str
    target_paper_id: str
    semantic_similarity: float = Field(ge=0, le=1)
    relationship_types: list[RelationshipType]
    relationships: list[RelationshipItem] = Field(min_length=1)
    summary: str = Field(min_length=3, max_length=2400)
    evidence_source: list[RelationshipEvidence] = Field(min_length=1)
    evidence_target: list[RelationshipEvidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source_analysis: AnalysisProvenance
    target_analysis: AnalysisProvenance
    relationship_provider: str
    relationship_model: str
    relationship_pipeline_version: str
    created_at: datetime
    cached: bool = False
    diagnostics: list[RelationshipDiagnostic] = Field(default_factory=list)

    _safe_summary = field_validator("summary")(_reject_html)


class RelationshipReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: ReviewDecision
    edited_relationship_types: list[RelationshipType] | None = Field(
        default=None, min_length=1, max_length=len(RelationshipType)
    )
    edited_summary: str | None = Field(default=None, min_length=3, max_length=2400)
    reviewer_note: str | None = Field(default=None, max_length=2000)

    @field_validator("edited_summary", "reviewer_note")
    @classmethod
    def reject_html(cls, value: str | None) -> str | None:
        return None if value is None else _reject_html(value)

    @model_validator(mode="after")
    def validate_edit_shape(self) -> RelationshipReviewRequest:
        has_edit = self.edited_relationship_types is not None or self.edited_summary is not None
        if self.decision == ReviewDecision.EDITED and not has_edit:
            raise ValueError("edited reviews require an edited summary or relationship types")
        if self.decision != ReviewDecision.EDITED and has_edit:
            raise ValueError("only edited reviews may contain edited values")
        return self


class RelationshipReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    proposal_id: str
    review_version: int = Field(ge=1)
    decision: ReviewDecision
    original_relationship_types: list[RelationshipType]
    original_summary: str
    relationship_pipeline_version: str
    edited_relationship_types: list[RelationshipType] | None = None
    edited_summary: str | None = None
    reviewer_note: str | None = None
    created_at: datetime


class RelationshipReviewHistory(BaseModel):
    proposal_id: str
    proposal: RelationshipProposal
    reviews: list[RelationshipReview]


class RelationshipErrorDetail(BaseModel):
    code: str
    message: str
