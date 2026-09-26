import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nadi9.domain.models import EpisodeLine, EvidenceRecord


class JSONLIngestionError(ValueError):
    """Raised when an evidence or episode JSONL file cannot be ingested."""


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load raw JSON objects from a JSONL file."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise JSONLIngestionError(
            f"Path is not a file: {file_path}"
        )

    records: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JSONLIngestionError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise JSONLIngestionError(
                    f"Line {line_number} must contain a JSON object."
                )

            records.append(record)

    return records


def ingest_evidence(path: str | Path) -> list[EvidenceRecord]:
    """Load and validate evidence records from a JSONL file."""

    raw_records = load_jsonl_records(path)
    evidence: list[EvidenceRecord] = []

    for index, record in enumerate(raw_records, start=1):
        try:
            evidence.append(EvidenceRecord.model_validate(record))
        except ValidationError as exc:
            raise JSONLIngestionError(
                f"Invalid evidence record at JSONL record {index}: "
                f"{exc}"
            ) from exc

    return evidence


def ingest_episode_lines(path: str | Path) -> list[EpisodeLine]:
    """Load and validate episode lines from a JSONL file."""

    raw_records = load_jsonl_records(path)
    episode_lines: list[EpisodeLine] = []

    for index, record in enumerate(raw_records, start=1):
        try:
            episode_lines.append(EpisodeLine.model_validate(record))
        except ValidationError as exc:
            raise JSONLIngestionError(
                f"Invalid episode record at JSONL record {index}: "
                f"{exc}"
            ) from exc

    return episode_lines