import json
import pytest

from nadi9.domain.enums import DecisionStatus
from nadi9.processor import EpisodeProcessor
from nadi9.providers import MockLLMProvider
from nadi9.storage import (
    AuditRepository,
    DecisionRepository,
    ReviewRepository,
    RunRepository,
    create_db_engine,
    create_session_factory,
    init_db,
)


@pytest.fixture
def db_session(tmp_path):
    db_file = tmp_path / "test_persistence.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_db_engine(db_url)
    init_db(engine)
    session_factory = create_session_factory(engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def make_episode_dict(subtitle_id: str, source_text: str) -> dict:
    return {
        "subtitle_id": subtitle_id,
        "source_text": source_text,
        "speaker": "Speaker A",
        "scene_id": "SCENE-01",
        "start_time": 0.0,
        "end_time": 2.0,
    }


def make_evidence_dict(evidence_id: str, content: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_id": "SRC-001",
        "evidence_type": "dictionary_a",
        "content": content,
        "provenance": {
            "original_file": "dict.json",
            "location": "loc-1",
            "extraction_method": "manual",
        },
    }


def test_episode_processor_with_database_persistence(tmp_path, db_session):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"

    episode_file.write_text(
        json.dumps(make_episode_dict("SUB-001", "formal greeting")) + "\n",
        encoding="utf-8",
    )
    evidence_file.write_text(
        json.dumps(make_evidence_dict("E-001", "formal greeting used by elders")) + "\n",
        encoding="utf-8",
    )

    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            }
        ]
    )

    processor = EpisodeProcessor(provider=provider, db_session=db_session)
    report = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
        run_id="run-persist-001",
        episode_id="ep-001",
    )

    assert report.run_id == "run-persist-001"
    assert report.accepted_count == 1

    # Verify database state
    run_repo = RunRepository(db_session)
    dec_repo = DecisionRepository(db_session)
    audit_repo = AuditRepository(db_session)

    run = run_repo.get_run("run-persist-001")
    assert run is not None
    assert run.status == "COMPLETED"

    decisions = dec_repo.list_decisions_for_run("run-persist-001")
    assert len(decisions) == 1
    assert decisions[0].subtitle_id == "SUB-001"
    assert decisions[0].nadi9_text == "Namaskar."

    events = audit_repo.list_events_for_run("run-persist-001")
    assert len(events) >= 2
    event_types = [e.event_type for e in events]
    assert "run_started" in event_types
    assert "run_completed" in event_types


def test_processor_idempotency_prevents_duplicate_processing(tmp_path, db_session):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"

    episode_file.write_text(
        json.dumps(make_episode_dict("SUB-001", "formal greeting")) + "\n",
        encoding="utf-8",
    )
    evidence_file.write_text(
        json.dumps(make_evidence_dict("E-001", "formal greeting used by elders")) + "\n",
        encoding="utf-8",
    )

    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            }
        ]
    )

    processor = EpisodeProcessor(provider=provider, db_session=db_session)

    # First run
    report1 = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
        run_id="run-idempotent-001",
    )
    assert report1.accepted_count == 1

    # Second run with same run_id (re-processing)
    report2 = processor.process_files(
        episode_path=episode_file,
        evidence_path=evidence_file,
        run_id="run-idempotent-001",
    )

    # Provider call count should remain 1 (skipped line processing due to idempotency)
    assert provider.call_count == 1

    dec_repo = DecisionRepository(db_session)
    decisions = dec_repo.list_decisions_for_run("run-idempotent-001")
    assert len(decisions) == 1
