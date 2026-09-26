from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration management using environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "testing", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("NADI9_ENV", "ENV"),
    )
    llm_provider: str = Field(
        default="mock",
        validation_alias=AliasChoices("NADI9_LLM_PROVIDER", "LLM_PROVIDER"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NADI9_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("NADI9_OPENAI_MODEL", "OPENAI_MODEL"),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NADI9_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )
    openai_timeout: float = Field(
        default=30.0,
        validation_alias=AliasChoices("NADI9_OPENAI_TIMEOUT", "OPENAI_TIMEOUT"),
    )

    database_url: str = Field(
        default="sqlite:///./nadi9.db",
        validation_alias=AliasChoices("NADI9_DATABASE_URL", "DATABASE_URL"),
    )
    checkpoint_db_path: str = Field(
        default="nadi9_checkpoints.db",
        validation_alias=AliasChoices("NADI9_CHECKPOINT_DB_PATH", "CHECKPOINT_DB_PATH"),
    )

    max_model_calls: int = Field(
        default=25,
        validation_alias=AliasChoices("NADI9_MAX_MODEL_CALLS", "MAX_MODEL_CALLS"),
    )
    max_tool_calls: int = Field(
        default=50,
        validation_alias=AliasChoices("NADI9_MAX_TOOL_CALLS", "MAX_TOOL_CALLS"),
    )
    retrieval_top_k: int = Field(
        default=5,
        validation_alias=AliasChoices("NADI9_RETRIEVAL_TOP_K", "RETRIEVAL_TOP_K"),
    )

    output_dir: Path = Field(
        default=Path("outputs"),
        validation_alias=AliasChoices("NADI9_OUTPUT_DIR", "OUTPUT_DIR"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("NADI9_LOG_LEVEL", "LOG_LEVEL"),
    )


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
