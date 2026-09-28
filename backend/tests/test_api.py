from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.api import ProviderHealth, SearchRequest, SearchResponse
from app.models.paper import Paper
from app.services.search_service import AllProvidersFailedError


class FailingSearchService:
    async def search(self, *_: object) -> None:
        raise AllProvidersFailedError("All scholarly providers are currently unavailable.")


class SuccessfulSearchService:
    async def search(self, request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            query=request.query,
            candidate_count=2,
            deduplicated_count=1,
            ranked_count=1,
            papers=[
                Paper(
                    id="paper-1",
                    title="Agentic Visualization",
                    publication_year=2024,
                    source_names=["openalex", "semantic_scholar"],
                    semantic_score=0.75,
                    reranker_score=2.5,
                )
            ],
            provider_status={
                "openalex": ProviderHealth(status="ok", successful_requests=4, failed_requests=0),
                "semantic_scholar": ProviderHealth(
                    status="degraded",
                    successful_requests=3,
                    failed_requests=1,
                    message="Some query variants failed; partial results remain available.",
                ),
                "arxiv": ProviderHealth(
                    status="ok", successful_requests=4, failed_requests=0, cached_requests=4
                ),
            },
        )


def test_health_and_search_validation(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        bad_limit = client.post("/api/search", json={"query": "agents", "limit": 2})
        assert bad_limit.status_code == 422

        bad_years = client.post(
            "/api/search",
            json={"query": "agents", "start_year": 2025, "end_year": 2020, "limit": 5},
        )
        assert bad_years.status_code == 422

        blank_query = client.post("/api/search", json={"query": " ", "limit": 5})
        assert blank_query.status_code == 422


def test_successful_search_returns_expected_response_shape(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.search_service = SuccessfulSearchService()
        response = client.post(
            "/api/search",
            json={"query": "visual agents", "start_year": 2020, "end_year": 2026, "limit": 5},
        )

    assert response.status_code == 200
    assert response.json() == {
        "query": "visual agents",
        "candidate_count": 2,
        "deduplicated_count": 1,
        "ranked_count": 1,
        "papers": [
            {
                "id": "paper-1",
                "title": "Agentic Visualization",
                "abstract": None,
                "authors": [],
                "publication_year": 2024,
                "publication_date": None,
                "venue": None,
                "doi": None,
                "arxiv_id": None,
                "arxiv_ids": [],
                "openalex_id": None,
                "semantic_scholar_id": None,
                "url": None,
                "alternate_urls": [],
                "pdf_url": None,
                "alternate_pdf_urls": [],
                "citation_count": None,
                "source_names": ["openalex", "semantic_scholar"],
                "semantic_score": 0.75,
                "reranker_score": 2.5,
            }
        ],
        "provider_status": {
            "openalex": {
                "status": "ok",
                "successful_requests": 4,
                "failed_requests": 0,
                "cached_requests": 0,
                "message": None,
            },
            "semantic_scholar": {
                "status": "degraded",
                "successful_requests": 3,
                "failed_requests": 1,
                "cached_requests": 0,
                "message": "Some query variants failed; partial results remain available.",
            },
            "arxiv": {
                "status": "ok",
                "successful_requests": 4,
                "failed_requests": 0,
                "cached_requests": 4,
                "message": None,
            },
        },
        "warnings": [],
    }


def test_all_provider_failure_returns_503(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.search_service = FailingSearchService()
        response = client.post("/api/search", json={"query": "agents", "limit": 5})
    assert response.status_code == 503
    assert response.json() == {"detail": "All scholarly providers are currently unavailable."}
