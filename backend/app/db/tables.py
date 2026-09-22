from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class SearchCacheEntry(Base):
    __tablename__ = "search_cache"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "query_key",
            "start_year_key",
            "end_year_key",
            "retrieval_limit",
            name="uq_search_cache",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    query_key: Mapped[str] = mapped_column(String(500))
    start_year_key: Mapped[str] = mapped_column(String(8), default="")
    end_year_key: Mapped[str] = mapped_column(String(8), default="")
    retrieval_limit: Mapped[int] = mapped_column(Integer)
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
