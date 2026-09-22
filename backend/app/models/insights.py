from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import ParsedPaper

INSIGHT_FIELDS = (
    "research_problem",
    "methods",
    "key_contributions",
    "main_findings",
    "why_it_matters",
    "target_audience",
    "limitations",
    "future_work",
)


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=500)


class InsightClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim: str = Field(min_length=1, max_length=500)
    evidence: list[EvidenceReference] = Field(min_length=1)


class InsightFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_problem: list[InsightClaim]
    methods: list[InsightClaim]
    key_contributions: list[InsightClaim]
    main_findings: list[InsightClaim]
    why_it_matters: list[InsightClaim]
    target_audience: list[InsightClaim]
    limitations: list[InsightClaim]
    future_work: list[InsightClaim]

    @classmethod
    def empty(cls) -> InsightFields:
        return cls(**{field: [] for field in INSIGHT_FIELDS})


class InsightsRequest(BaseModel):
    document: ParsedPaper


class InsightsResponse(BaseModel):
    paper_id: str
    document_fingerprint: str
    model: str
    extraction_version: str
    cached: bool
    insights: InsightFields
