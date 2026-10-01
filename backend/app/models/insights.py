from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import ParsedPaper

INSIGHT_FIELDS = (
    "paper_overview",
    "research_problem",
    "methods",
    "key_contributions",
    "evaluation",
    "main_findings",
    "why_it_matters",
    "target_audience",
    "limitations",
    "future_work",
)
EVIDENCE_QUOTE_MAX_LENGTH = 500
PAPER_OVERVIEW_MAX_LENGTH = 6000
INSIGHT_CLAIM_MAX_LENGTH = 6000


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=EVIDENCE_QUOTE_MAX_LENGTH)


class InsightClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim: str = Field(min_length=1, max_length=INSIGHT_CLAIM_MAX_LENGTH)
    # Markdown providers are encouraged to cite source chunks, but a useful
    # paraphrase is not discarded merely because a provider omitted a citation.
    evidence: list[EvidenceReference] = Field(default_factory=list)


class InsightFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_overview: list[InsightClaim] = Field(default_factory=list)
    research_problem: list[InsightClaim] = Field(default_factory=list)
    methods: list[InsightClaim] = Field(default_factory=list)
    key_contributions: list[InsightClaim] = Field(default_factory=list)
    evaluation: list[InsightClaim] = Field(default_factory=list)
    main_findings: list[InsightClaim] = Field(default_factory=list)
    why_it_matters: list[InsightClaim] = Field(default_factory=list)
    target_audience: list[InsightClaim] = Field(default_factory=list)
    limitations: list[InsightClaim] = Field(default_factory=list)
    future_work: list[InsightClaim] = Field(default_factory=list)

    @classmethod
    def empty(cls) -> InsightFields:
        return cls(**{field: [] for field in INSIGHT_FIELDS})


class InsightsRequest(BaseModel):
    document: ParsedPaper
    provider: str | None = Field(
        default=None, min_length=1, max_length=40, pattern=r"^[a-z0-9_-]+$"
    )


class InsightsResponse(BaseModel):
    paper_id: str
    document_fingerprint: str
    model: str
    extraction_version: str
    cached: bool
    insights: InsightFields


EvidenceCategory = Literal[
    "problem",
    "approach_method",
    "contribution",
    "evaluation",
    "finding",
    "significance",
    "audience",
    "limitation",
    "future_work",
]


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str = Field(min_length=1, max_length=40)
    category: EvidenceCategory
    claim: str = Field(min_length=1, max_length=500)
    evidence_refs: list[EvidenceReference] = Field(min_length=1)
    section_id: str | None = None
    explicitness: Literal["explicit", "synthesized"] = "explicit"
    provider: str = Field(min_length=1, max_length=40)


class ValidatedEvidenceLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[EvidenceClaim] = Field(default_factory=list)


class LLMProviderStatus(BaseModel):
    provider_id: str
    model: str
    configured: bool
    cloud: bool
    availability: Literal[
        "configured",
        "unconfigured",
        "available",
        "temporarily_unavailable",
        "authentication_error",
    ]
    message: str


class LLMProviderStatusResponse(BaseModel):
    default_provider: str
    providers: list[LLMProviderStatus]
