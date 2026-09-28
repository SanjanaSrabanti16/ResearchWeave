from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RANKING_PROFILES = {
    "fast": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "rerank_shortlist_size": 100,
        "ranking_fusion_mode": "rrf",
    },
    "quality": {
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "rerank_shortlist_size": 100,
        "ranking_fusion_mode": "reranker",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    app_name: str = "ResearchWeave"
    semantic_scholar_api_key: str | None = None
    openalex_api_key: str | None = None
    ranking_profile: Literal["fast", "quality"] = "fast"
    embedding_model: str
    reranker_model: str
    database_url: str = f"sqlite:///{Path(__file__).parents[2] / 'data' / 'cache.sqlite3'}"
    cache_ttl_hours: int = Field(default=24, ge=1, le=720)
    candidate_target: int = Field(default=240, ge=30, le=300)
    rerank_shortlist_size: int = Field(ge=10, le=300)
    ranking_fusion_mode: Literal["reranker", "rrf"]
    provider_timeout_seconds: float = Field(default=20.0, ge=1, le=120)
    provider_retries: int = Field(default=2, ge=0, le=5)
    cors_origins: str = "http://localhost:5173"
    unpaywall_email: str | None = None
    pdf_max_size_mb: int = Field(default=50, ge=1, le=200)
    pdf_download_timeout_seconds: float = Field(default=30.0, ge=1, le=120)
    pdf_allowed_hosts: str = (
        "arxiv.org,biorxiv.org,core.ac.uk,europepmc.org,hal.science,medrxiv.org,"
        "ncbi.nlm.nih.gov,openreview.net,osf.io,semanticscholar.org,zenodo.org"
    )
    grobid_url: str = "http://localhost:8070"
    grobid_timeout_seconds: float = Field(default=120.0, ge=5, le=600)
    grobid_parser_version: str = "0.9.1-crf+frontmatter-v1"
    parsed_document_cache_dir: str = str(Path(__file__).parents[2] / "data" / "parsed_documents")
    ollama_model: str = "qwen3:1.7b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = Field(default=180.0, ge=10, le=1800)
    ollama_batch_chars: int = Field(default=13000, ge=2000, le=30000)
    llm_provider: str = "ollama"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash"
    gemini_timeout_seconds: float = Field(default=180.0, ge=10, le=1800)
    evl_gemma_api_key: SecretStr | None = None
    evl_gemma_base_url: str = "https://sage200.evl.uic.edu"
    evl_gemma_model: str = "gemma4"
    evl_gemma_timeout_seconds: float = Field(default=180.0, ge=10, le=1800)
    llm_diagnostic_dir: str = str(Path(__file__).parents[2] / "data" / "llm_diagnostics")
    insight_cache_dir: str = str(Path(__file__).parents[2] / "data" / "insights")

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"ollama", "gemini", "evl_gemma"}:
            raise ValueError("LLM_PROVIDER must be ollama, gemini, or evl_gemma")
        return normalized

    @field_validator("evl_gemma_base_url")
    @classmethod
    def require_https_evl_gemma(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("EVL_GEMMA_BASE_URL must be an HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "EVL_GEMMA_BASE_URL must not include credentials, a query, or a fragment"
            )
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def require_local_ollama(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
            "host.docker.internal",
        }:
            raise ValueError("OLLAMA_BASE_URL must be a local HTTP Ollama address")
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path
            not in {
                "",
                "/",
            }
        ):
            raise ValueError("OLLAMA_BASE_URL must not include credentials, path, or query")
        return value.rstrip("/")

    @model_validator(mode="before")
    @classmethod
    def apply_ranking_profile(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        profile = values.get("ranking_profile", "fast")
        defaults = RANKING_PROFILES.get(profile)
        if defaults:
            values = {**defaults, **values}
        return values

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def pdf_allowed_host_list(self) -> tuple[str, ...]:
        return tuple(
            host.strip().casefold() for host in self.pdf_allowed_hosts.split(",") if host.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
