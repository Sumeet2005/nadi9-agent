import sqlite3
import pytest

from nadi9.domain.enums import DecisionStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Provenance
from nadi9.graph.workflow import build_nadi9_workflow
from nadi9.processor import EpisodeProcessor
from nadi9.providers.mock import MockLLMProvider
from nadi9.storage import (
    AuditRepository,
    DecisionRepository,
    ReviewRepository,
    RunRepository,
    SqliteCheckpointSaver,
    create_db_engine,
    create_session_factory,
    init_db,
)


def test_end_to_end_process_restart_and_hitl_recovery(tmp_path):
    """Full end-to-end integration test proving process restart recovery, zero redundant LLM calls, DB persistence, and audit log tracking."""

    db_file = tmp_path / "e2e_app.db"
    chk_file = str(tmp_path / "e2e_checkpoints.db")

    engine = create_db_engine(f"sqlite:///{db_file}")
    init_db(engine)
    session_factory = create_session_factory(engine)

    # Configured Mock LLM Provider
    mock_provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-E2E-01",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001", "E-002"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )

    episode_line = EpisodeLine(
        subtitle_id="SUB-E2E-01",
        source_text="formal greeting",
        speaker="Speaker A",
        scene_id="SCENE-E2E-01",
        start_time=0.0,
        end_time=2.0,
    )

    evidence_records = [
        EvidenceRecord(
            evidence_id="E-001",
            source_id="SRC-1",
            evidence_type="dictionary_a",
            content="formal greeting used by elders",
            metadata={"conflicts_with": ["E-002"]},
            provenance=Provenance(original_file="d.json", extraction_method="manual"),
        ),
        EvidenceRecord(
            evidence_id="E-002",
            source_id="SRC-1",
            evidence_type="dictionary_b",
            content="informal greeting",
            provenance=Provenance(original_file="d.json", extraction_method="manual"),
        ),
    ]

    # --- Step 1: Initialize Process Instance A ---
    session_a = session_factory()
    saver_a = SqliteCheckpointSaver(chk_file)
    workflow_a = build_nadi9_workflow(mock_provider, checkpointer=saver_a)

    processor_a = EpisodeProcessor(
        provider=mock_provider,
        db_session=session_a,
        workflow=workflow_a,
    )

    # 1. Start processing -> 2. Generate hypothesis -> 3. Verify -> 4. Conflict detected -> 5. Interrupt & 6. Persist Checkpoint
    report_a = processor_a.process_records(
        episode_lines=[episode_line],
        evidence=evidence_records,
        run_id="run-e2e-101",
        episode_id="ep-e2e-101",
    )

    assert report_a.human_review_count == 1
    calls_before_restart = mock_provider.call_count

    # Verify initial database state in SQLite before restart
    run_repo_a = RunRepository(session_a)
    run_db = run_repo_a.get_run("run-e2e-101")
    assert run_db is not None
    assert run_db.status == "WAITING_FOR_REVIEW"

    rev_repo_a = ReviewRepository(session_a)
    pending_reviews_a = rev_repo_a.list_review_items_for_run("run-e2e-101", pending_only=True)
    assert len(pending_reviews_a) == 1
    review_id = pending_reviews_a[0].review_id

    session_a.close()

    # --- Step 7: Simulate Complete Process Restart (Destroy Instance A) ---
    del processor_a
    del workflow_a
    del saver_a

    # --- Step 8: Create New Process Instance B & Step 9: Load SQLite Checkpoint ---
    session_b = session_factory()
    saver_b = SqliteCheckpointSaver(chk_file)
    workflow_b = build_nadi9_workflow(mock_provider, checkpointer=saver_b)

    processor_b = EpisodeProcessor(
        provider=mock_provider,
        db_session=session_b,
        workflow=workflow_b,
    )

    # --- Step 10 & 11: Human Correct Action & Resume via Command(resume=...) ---
    final_decision = processor_b.resume_human_review(
        run_id="run-e2e-101",
        subtitle_id="SUB-E2E-01",
        action_type="correct",
        text="Namaste, Ji!",
        reason="Respectful register correction.",
        actor="lead_linguist",
    )

    # --- Step 12: Verify Workflow Completed ---
    assert final_decision.status == DecisionStatus.ACCEPTED
    assert final_decision.human_review_required is False

    # --- Step 13: Verify No Second LLM Generation Occurred ---
    calls_after_resume = mock_provider.call_count
    assert calls_after_resume == calls_before_restart

    # --- Step 14: Verify DB Persistence ---
    run_repo_b = RunRepository(session_b)
    completed_run = run_repo_b.get_run("run-e2e-101")
    assert completed_run.status == "COMPLETED"

    dec_repo_b = DecisionRepository(session_b)
    saved_decs = dec_repo_b.list_decisions_for_run("run-e2e-101")
    assert len(saved_decs) == 1
    assert saved_decs[0].nadi9_text == "Namaste, Ji!"

    # --- Step 15: Verify Audit Trail ---
    audit_repo_b = AuditRepository(session_b)
    audit_events = audit_repo_b.list_events_for_run("run-e2e-101")
    assert any(e.event_type == "human_review_corrected" for e in audit_events)

    # --- Step 16: Verify Final Decision Text ---
    assert final_decision.nadi9_text == "Namaste, Ji!"

    # --- Step 17: Verify Review Item Resolved ---
    rev_repo_b = ReviewRepository(session_b)
    pending_after = rev_repo_b.list_review_items_for_run("run-e2e-101", pending_only=True)
    assert len(pending_after) == 0

    session_b.close()
