from collections.abc import Generator
from typing import Any

from sqlalchemy.orm import Session

from nadi9.config import get_settings
from nadi9.providers.mock import MockLLMProvider
from nadi9.storage.database import create_db_engine, create_session_factory, init_db

_engine = None
_session_factory = None


def get_engine() -> Any:
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = create_db_engine(settings.database_url)
        init_db(_engine)
        _session_factory = create_session_factory(_engine)
    return _engine


def get_db() -> Generator[Session, None, None]:
    get_engine()
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()
