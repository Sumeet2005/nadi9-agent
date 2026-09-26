import logging
import sys
from typing import Any

from nadi9.config import get_settings


def configure_logging(level: str | None = None) -> None:
    """Configure system-wide structured logging."""
    settings = get_settings()
    log_level_str = level or settings.log_level

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    root_logger = logging.getLogger("nadi9")
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_logger(name: str = "nadi9") -> logging.Logger:
    """Return a contextual logger instance."""
    logger = logging.getLogger(f"nadi9.{name}")
    if not logging.getLogger("nadi9").handlers:
        configure_logging()
    return logger


def sanitize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Sanitize secrets, bearer tokens, and API keys from log fields."""
    sanitized = {}
    for key, val in kwargs.items():
        k_lower = key.lower()
        if any(secret_term in k_lower for secret_term in ("key", "secret", "token", "bearer", "auth", "password")):
            sanitized[key] = "***MASKED***"
        else:
            sanitized[key] = val
    return sanitized


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    run_id: str | None = None,
    episode_id: str | None = None,
    subtitle_id: str | None = None,
    thread_id: str | None = None,
    current_step: str | None = None,
    duration_ms: float | None = None,
    status: str | None = None,
    **kwargs: Any,
) -> None:
    """Log a structured contextual event with secret masking."""
    sanitized = sanitize_kwargs(kwargs)
    ctx_parts = []
    if run_id:
        ctx_parts.append(f"run_id={run_id}")
    if episode_id:
        ctx_parts.append(f"episode_id={episode_id}")
    if subtitle_id:
        ctx_parts.append(f"subtitle_id={subtitle_id}")
    if thread_id:
        ctx_parts.append(f"thread_id={thread_id}")
    if current_step:
        ctx_parts.append(f"step={current_step}")
    if status:
        ctx_parts.append(f"status={status}")
    if duration_ms is not None:
        ctx_parts.append(f"duration_ms={duration_ms:.2f}")

    extra_str = " ".join(f"{k}={v}" for k, v in sanitized.items())
    if extra_str:
        ctx_parts.append(extra_str)

    context_str = f" [{', '.join(ctx_parts)}]" if ctx_parts else ""
    logger.info(f"{event}{context_str}")
