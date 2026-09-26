import json

import pytest

from nadi9.ingestion import (
    JSONLIngestionError,
    ingest_episode_lines,
    ingest_evidence,
    load_jsonl_records,
)


def make_evidence_record(
    evidence_id: str,
    content: str,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_id": "SRC-001",
        "evidence_type": "dictionary_a",
        "content": content,
        "provenance": {
            "original_file": "dictionary.json",
            "location": "entry-1",
            "extraction_method": "manual",
            "source_reference": "SRC-001",
        },
    }


def make_episode_record(
    subtitle_id: str = "SUB-001",
    source_text: str = "Hello",
) -> dict:
    return {
        "subtitle_id": subtitle_id,
        "source_text": source_text,
        "speaker": "Speaker 1",
        "scene_id": "SCENE-01",
        "start_time": 0.0,
        "end_time": 2.5,
    }


def test_load_jsonl_records_reads_json_objects(tmp_path):
    path = tmp_path / "evidence.jsonl"

    records = [
        make_evidence_record("E-001", "first"),
        make_evidence_record("E-002", "second"),
    ]

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    result = load_jsonl_records(path)

    assert len(result) == 2
    assert result[0]["evidence_id"] == "E-001"
    assert result[1]["evidence_id"] == "E-002"


def test_load_jsonl_records_ignores_blank_lines(tmp_path):
    path = tmp_path / "evidence.jsonl"

    record = make_evidence_record("E-001", "first")

    path.write_text(
        f"\n{json.dumps(record)}\n\n",
        encoding="utf-8",
    )

    result = load_jsonl_records(path)

    assert len(result) == 1


def test_ingest_evidence_validates_records(tmp_path):
    path = tmp_path / "evidence.jsonl"

    records = [
        make_evidence_record("E-001", "first"),
        make_evidence_record("E-002", "second"),
    ]

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    result = ingest_evidence(path)

    assert len(result) == 2
    assert result[0].evidence_id == "E-001"
    assert result[0].content == "first"
    assert result[1].evidence_id == "E-002"


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '{"evidence_id": "E-001"}\n'
        '{"broken": ',
        encoding="utf-8",
    )

    with pytest.raises(
        JSONLIngestionError,
        match="Invalid JSON on line 2",
    ):
        load_jsonl_records(path)


def test_non_object_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '["not", "an", "object"]',
        encoding="utf-8",
    )

    with pytest.raises(
        JSONLIngestionError,
        match="must contain a JSON object",
    ):
        load_jsonl_records(path)


def test_invalid_evidence_schema_is_rejected(tmp_path):
    path = tmp_path / "invalid_evidence.jsonl"

    invalid_record = make_evidence_record(
        "E-001",
        "",
    )

    path.write_text(
        json.dumps(invalid_record),
        encoding="utf-8",
    )

    with pytest.raises(
        JSONLIngestionError,
        match="Invalid evidence record",
    ):
        ingest_evidence(path)


def test_missing_file_is_rejected(tmp_path):
    path = tmp_path / "does-not-exist.jsonl"

    with pytest.raises(FileNotFoundError):
        load_jsonl_records(path)


def test_ingest_episode_lines_validates_records(tmp_path):
    path = tmp_path / "episodes.jsonl"

    records = [
        make_episode_record("SUB-001", "Hello"),
        make_episode_record("SUB-002", "Goodbye"),
    ]

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    result = ingest_episode_lines(path)

    assert len(result) == 2
    assert result[0].subtitle_id == "SUB-001"
    assert result[0].source_text == "Hello"
    assert result[1].subtitle_id == "SUB-002"


def test_invalid_episode_schema_is_rejected(tmp_path):
    path = tmp_path / "invalid_episodes.jsonl"

    invalid_record = make_episode_record("SUB-001", "")  # Empty source text

    path.write_text(
        json.dumps(invalid_record),
        encoding="utf-8",
    )

    with pytest.raises(
        JSONLIngestionError,
        match="Invalid episode record at JSONL record 1",
    ):
        ingest_episode_lines(path)