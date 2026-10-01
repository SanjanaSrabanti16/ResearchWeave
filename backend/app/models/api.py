from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.document import ParsedPaper
from app.models.insights import InsightsResponse
from app.models.paper import Paper


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=500)
    start_year: int | None = Field(default=None, ge=1800, le=2100)
    end_year: int | None = Field(default=None, ge=1800, le=2100)
    limit: int = Field(default=20, ge=5, le=50)

    @model_validator(mode="after")
    def validate_year_range(self) -> "SearchRequest":
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        return self


class ProviderHealth(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    successful_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    cached_requests: int = Field(default=0, ge=0)
    message: str | None = None


class SearchResponse(BaseModel):
    query: str
    overall_status: Literal["success", "no_results"] = "success"
    candidate_count: int
    deduplicated_count: int
    ranked_count: int
    papers: list[Paper]
    provider_status: dict[str, ProviderHealth]
    warnings: list[str] = Field(default_factory=list)


class GraphRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=500)
    papers: list[Paper] = Field(min_length=1, max_length=50)


class CachedPaperAnalysisResponse(BaseModel):
    paper_id: str
    document: ParsedPaper | None = None
    insights: InsightsResponse | None = None
    insight_provider: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    application: str
    version: str
    timestamp: datetime
