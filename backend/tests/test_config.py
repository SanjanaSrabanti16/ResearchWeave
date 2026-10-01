import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm import EVLGemmaProvider, LLMProviderRegistry


def test_fast_ranking_profile_is_default() -> None:
    settings = Settings(_env_file=None)
    assert settings.ranking_profile == "fast"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert settings.rerank_shortlist_size == 100
    assert settings.ranking_fusion_mode == "rrf"


def test_quality_ranking_profile_configuration() -> None:
    settings = Settings(_env_file=None, ranking_profile="quality")
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.rerank_shortlist_size == 100
    assert settings.ranking_fusion_mode == "reranker"


def test_explicit_ranking_settings_override_profile_defaults() -> None:
    settings = Settings(
        _env_file=None,
        ranking_profile="quality",
        embedding_model="custom-embedding",
        reranker_model="custom-reranker",
        rerank_shortlist_size=37,
        ranking_fusion_mode="rrf",
    )
    assert settings.embedding_model == "custom-embedding"
    assert settings.reranker_model == "custom-reranker"
    assert settings.rerank_shortlist_size == 37
    assert settings.ranking_fusion_mode == "rrf"


def test_environment_overrides_win_over_profile_defaults(monkeypatch) -> None:
    monkeypatch.setenv("RANKING_PROFILE", "quality")
    monkeypatch.setenv("EMBEDDING_MODEL", "environment-embedding")
    monkeypatch.setenv("RERANK_SHORTLIST_SIZE", "41")
    settings = Settings(_env_file=None)
    assert settings.ranking_profile == "quality"
    assert settings.embedding_model == "environment-embedding"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.rerank_shortlist_size == 41


def test_candidate_target_is_capped_at_300() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, candidate_target=301)


def test_evl_environment_settings_load_into_provider_registry(monkeypatch) -> None:
    monkeypatch.setenv("EVL_GEMMA_API_KEY", "test-key")
    monkeypatch.setenv("EVL_GEMMA_BASE_URL", "https://evl.example/v1")
    monkeypatch.setenv("EVL_GEMMA_MODEL", "gemma4")
    settings = Settings(_env_file=None)
    provider = EVLGemmaProvider(
        api_key=settings.evl_gemma_api_key.get_secret_value(),
        base_url=settings.evl_gemma_base_url,
        model_id=settings.evl_gemma_model,
    )
    registry = LLMProviderRegistry([provider], default_provider="evl_gemma")

    status = registry.statuses()[0]
    assert status == {
        "provider_id": "evl_gemma",
        "model": "gemma4",
        "configured": True,
        "cloud": True,
        "availability": "configured",
        "message": "Configured; availability has not been checked yet",
    }
    assert "test-key" not in str(status)
