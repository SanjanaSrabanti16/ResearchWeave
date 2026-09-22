from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class SearchResponse(BaseModel):
    query: str
    candidate_count: int
    deduplicated_count: int
    ranked_count: int
    papers: list[Paper]
    provider_status: dict[str, str]
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    application: str
    version: str
    timestamp: datetime
