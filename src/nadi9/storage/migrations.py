from typing import Any

from .database import init_db


def run_migrations(engine: Any) -> None:
    """Run idempotent schema initialization/migrations."""
    init_db(engine)
