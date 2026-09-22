import json
import re
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.tables import SearchCacheEntry
from app.models.paper import PaperCandidate


class CacheService:
    def __init__(self, session_factory: sessionmaker[Session], ttl_hours: int = 24) -> None:
        self.session_factory = session_factory
        self.ttl = timedelta(hours=ttl_hours)

    @staticmethod
    def _query_key(query: str) -> str:
        return re.sub(r"\s+", " ", query).strip().casefold()

    @staticmethod
    def _year_key(year: int | None) -> str:
        return "" if year is None else str(year)

    def get(
        self,
        provider: str,
        query: str,
        start_year: int | None,
        end_year: int | None,
        retrieval_limit: int,
    ) -> list[PaperCandidate] | None:
        with self.session_factory() as session:
            statement = select(SearchCacheEntry).where(
                SearchCacheEntry.provider == provider,
                SearchCacheEntry.query_key == self._query_key(query),
                SearchCacheEntry.start_year_key == self._year_key(start_year),
                SearchCacheEntry.end_year_key == self._year_key(end_year),
                SearchCacheEntry.retrieval_limit == retrieval_limit,
            )
            entry = session.scalar(statement)
            if entry is None:
                return None
            created_at = entry.created_at.replace(tzinfo=UTC)
            if datetime.now(UTC) - created_at > self.ttl:
                session.delete(entry)
                session.commit()
                return None
            return [PaperCandidate.model_validate(item) for item in json.loads(entry.payload)]

    def set(
        self,
        provider: str,
        query: str,
        start_year: int | None,
        end_year: int | None,
        retrieval_limit: int,
        papers: list[PaperCandidate],
    ) -> None:
        query_key = self._query_key(query)
        start_key = self._year_key(start_year)
        end_key = self._year_key(end_year)
        payload = json.dumps([paper.model_dump(mode="json") for paper in papers])
        with self.session_factory() as session:
            statement = select(SearchCacheEntry).where(
                SearchCacheEntry.provider == provider,
                SearchCacheEntry.query_key == query_key,
                SearchCacheEntry.start_year_key == start_key,
                SearchCacheEntry.end_year_key == end_key,
                SearchCacheEntry.retrieval_limit == retrieval_limit,
            )
            entry = session.scalar(statement)
            if entry is None:
                entry = SearchCacheEntry(
                    provider=provider,
                    query_key=query_key,
                    start_year_key=start_key,
                    end_year_key=end_key,
                    retrieval_limit=retrieval_limit,
                    payload=payload,
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                )
                session.add(entry)
            else:
                entry.payload = payload
                entry.created_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
