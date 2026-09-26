from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base ORM model for SQLAlchemy tables."""


def create_db_engine(db_url: str = "sqlite:///./nadi9.db") -> Any:
    """Create a SQLAlchemy Engine with sensible sqlite/pooling defaults."""
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(
        db_url,
        connect_args=connect_args,
        future=True,
    )


def create_session_factory(engine: Any) -> sessionmaker[Session]:
    """Create a sessionmaker factory bound to an engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(engine: Any) -> None:
    """Initialize database tables idempotently."""
    Base.metadata.create_all(bind=engine)
