from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    if database_url.startswith("sqlite:///"):
        database_path = database_url.removeprefix("sqlite:///")
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    )
    return sessionmaker(bind=engine, expire_on_commit=False)


def initialize_database(session_factory: sessionmaker[Session]) -> None:
    engine = session_factory.kw["bind"]
    inspector = inspect(engine)
    if "search_cache" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("search_cache")}
        if "retrieval_limit" not in columns:
            with engine.begin() as connection:
                connection.exec_driver_sql("DROP TABLE search_cache")
    Base.metadata.create_all(engine)
