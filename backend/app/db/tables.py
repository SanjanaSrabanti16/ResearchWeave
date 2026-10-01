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


class RelationshipProposalEntry(Base):
    __tablename__ = "relationship_proposals"

    proposal_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_paper_id: Mapped[str] = mapped_column(String(200), index=True)
    target_paper_id: Mapped[str] = mapped_column(String(200), index=True)
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class RelationshipReviewEntry(Base):
    __tablename__ = "relationship_reviews"
    __table_args__ = (
        UniqueConstraint(
            "proposal_id",
            "request_hash",
            name="uq_relationship_review_request",
        ),
    )

    review_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(String(36), index=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    review_version: Mapped[int] = mapped_column(Integer)
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
