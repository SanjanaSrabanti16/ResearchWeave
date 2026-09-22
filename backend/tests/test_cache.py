import sqlite3
from datetime import UTC, datetime, timedelta

from sqlalchemy import inspect, update

from app.db.database import create_session_factory, initialize_database
from app.db.tables import SearchCacheEntry
from app.models.paper import PaperCandidate
from app.services.cache_service import CacheService


def test_cache_round_trip_and_query_normalization(tmp_path) -> None:
    factory = create_session_factory(f"sqlite:///{tmp_path / 'cache.sqlite3'}")
    initialize_database(factory)
    cache = CacheService(factory, ttl_hours=24)
    papers = [PaperCandidate(title="A paper", source_names=["test"])]
    cache.set("test", "  Agent   Search ", 2020, 2025, 80, papers)
    cached = cache.get("test", "agent search", 2020, 2025, 80)
    assert cached is not None
    assert cached[0].title == "A paper"


def test_expired_cache_is_deleted(tmp_path) -> None:
    factory = create_session_factory(f"sqlite:///{tmp_path / 'cache.sqlite3'}")
    initialize_database(factory)
    cache = CacheService(factory, ttl_hours=1)
    cache.set("test", "query", None, None, 80, [PaperCandidate(title="Old")])
    with factory() as session:
        session.execute(
            update(SearchCacheEntry).values(
                created_at=(datetime.now(UTC) - timedelta(hours=2)).replace(tzinfo=None)
            )
        )
        session.commit()
    assert cache.get("test", "query", None, None, 80) is None


def test_cache_identity_includes_retrieval_limit(tmp_path) -> None:
    factory = create_session_factory(f"sqlite:///{tmp_path / 'cache.sqlite3'}")
    initialize_database(factory)
    cache = CacheService(factory)
    cache.set("test", "query", None, None, 20, [PaperCandidate(title="Twenty")])
    assert cache.get("test", "query", None, None, 20) is not None
    assert cache.get("test", "query", None, None, 100) is None


def test_old_cache_schema_is_reset_for_retrieval_limit(tmp_path) -> None:
    database_path = tmp_path / "old-cache.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE search_cache (id INTEGER PRIMARY KEY, provider VARCHAR(50))"
        )
    factory = create_session_factory(f"sqlite:///{database_path}")
    initialize_database(factory)
    columns = {column["name"] for column in inspect(factory.kw["bind"]).get_columns("search_cache")}
    assert "retrieval_limit" in columns
