from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ProcessRunRequest(BaseModel):
    run_id: str | None = None
    episode_id: str | None = None
    episode_lines: list[dict[str, Any]] = Field(default_factory=list)
    evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    episode_file_path: str | None = None
    evidence_file_path: str | None = None


class RunSummaryResponse(BaseModel):
    run_id: str
    episode_id: str
    status: str
    total_subtitles_processed: int
    accepted_count: int
    human_review_count: int
    created_at: str
    completed_at: str | None = None


class ReviewActionApiRequest(BaseModel):
    action_type: str | None = None
    actor: str = "human_reviewer"
    reason: str | None = None
    text: str | None = None
    affected_subtitle_ids: list[str] | None = None
