import sqlite3
import pytest

from nadi9.domain.enums import DecisionStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Provenance
from nadi9.graph.state import create_initial_state
from nadi9.graph.workflow import Nadi9Workflow, build_nadi9_workflow
from nadi9.processor import EpisodeProcessor
from nadi9.providers import MockLLMProvider
from nadi9.storage.checkpointer import (
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointNotFoundError,
    SqliteCheckpointSaver,
)


def make_conflicting_evidence() -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            evidence_id="E-001",
            source_id="SRC-001",
            evidence_type="dictionary_a",
            content="formal greeting",
            metadata={"conflicts_with": ["E-002"]},
            provenance=Provenance(
                original_file="dict.json",
                location="loc-1",
                extraction_method="manual",
            ),
        ),
        EvidenceRecord(
            evidence_id="E-002",
            source_id="SRC-001",
            evidence_type="dictionary_b",
            content="informal greeting",
            provenance=Provenance(
                original_file="dict.json",
                location="loc-2",
                extraction_method="manual",
            ),
        ),
    ]


def make_episode() -> EpisodeLine:
    return EpisodeLine(
        subtitle_id="SUB-DURABLE-01",
        source_text="formal greeting",
        speaker="Speaker A",
        scene_id="SCENE-01",
        start_time=0.0,
        end_time=2.0,
    )


def test_durable_checkpoint_process_restart_recovery(tmp_path):
    chk_db = str(tmp_path / "durable_checkpoints.db")
    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001", "E-002"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )

    # Workflow Instance A (simulating original process)
    saver_a = SqliteCheckpointSaver(chk_db)
    workflow_a = build_nadi9_workflow(provider, checkpointer=saver_a)
    state = create_initial_state("run-durable-1", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    thread_id = "run-durable-1:SUB-DURABLE-01"
    interrupted_state = workflow_a.run(state, thread_id=thread_id)

    assert interrupted_state["current_step"] == "human_review"
    calls_before_restart = provider.call_count

    # Simulate Process Restart: Destroy instance A objects completely
    del workflow_a
    del saver_a

    # Workflow Instance B (simulating new process created after restart)
    saver_b = SqliteCheckpointSaver(chk_db)
    workflow_b = build_nadi9_workflow(provider, checkpointer=saver_b)

    # Verify Instance B can load the persisted checkpoint from SQLite
    snapshot = workflow_b.get_state(thread_id)
    assert snapshot is not None
    assert snapshot.next == ("route_human_review",)

    action = {
        "action_type": "correct",
        "text": "Namaste, Ji!",
        "reason": "Corrected for respectful register",
        "actor": "editor_durable",
    }
    final_state = workflow_b.resume(thread_id, action)

    calls_after_resume = provider.call_count

    # Prove resume did NOT consume extra LLM calls
    assert calls_after_resume == calls_before_restart
    assert final_state["current_step"] == "finalize"

    decision = final_state["subtitle_decisions"][0]
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.nadi9_text == "Namaste, Ji!"
    assert len(final_state["human_review_actions"]) == 1
    assert final_state["human_review_actions"][0].original_nadi9_text == "Namaskar."


def test_missing_checkpoint_raises_clean_exception(tmp_path):
    chk_db = str(tmp_path / "missing_chk.db")
    saver = SqliteCheckpointSaver(chk_db)
    provider = MockLLMProvider(responses=[])
    workflow = build_nadi9_workflow(provider, checkpointer=saver)

    action = {"action_type": "approve", "actor": "editor"}
    with pytest.raises(CheckpointNotFoundError, match="No active execution checkpoint found"):
        workflow.resume("non_existent_thread_999", action)


def test_resume_after_completed_workflow_raises_exception(tmp_path):
    chk_db = str(tmp_path / "completed_chk.db")
    saver = SqliteCheckpointSaver(chk_db)
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
    workflow = build_nadi9_workflow(provider, checkpointer=saver)
    state = create_initial_state("run-comp-1", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = [
        EvidenceRecord(
            evidence_id="E-001",
            source_id="SRC-1",
            evidence_type="dictionary_a",
            content="formal greeting",
            provenance=Provenance(original_file="f.json", extraction_method="manual"),
        )
    ]

    thread_id = "run-comp-1:SUB-DURABLE-01"
    res = workflow.run(state, thread_id=thread_id)
    assert res["current_step"] == "finalize"

    action = {"action_type": "approve", "actor": "editor"}
    with pytest.raises(CheckpointError, match="already completed execution"):
        workflow.resume(thread_id, action)


def test_corrupted_checkpoint_raises_clean_exception(tmp_path):
    chk_db = str(tmp_path / "corrupt_chk.db")
    saver = SqliteCheckpointSaver(chk_db)

    # Insert malformed checkpoint blob directly into SQLite
    with sqlite3.connect(chk_db) as conn:
        conn.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("thread-corrupt", "", "chk-1", "corrupt_type", b"invalid_raw_bytes", "meta_type", b"meta_bytes", None),
        )

    with pytest.raises(CheckpointCorruptedError, match="Corrupted checkpoint blob"):
        saver.get_tuple({"configurable": {"thread_id": "thread-corrupt"}})


def test_episode_processor_with_durable_checkpointer(tmp_path):
    chk_db = str(tmp_path / "proc_chk.db")
    saver = SqliteCheckpointSaver(chk_db)
    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001", "E-002"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )

    processor = EpisodeProcessor(
        provider=provider,
        workflow=build_nadi9_workflow(provider, checkpointer=saver),
    )

    report = processor.process_records(
        episode_lines=[make_episode()],
        evidence=make_conflicting_evidence(),
        run_id="run-proc-durable-1",
    )
    assert report.human_review_count == 1

    final_dec = processor.resume_human_review(
        run_id="run-proc-durable-1",
        subtitle_id="SUB-DURABLE-01",
        action_type="approve",
        actor="editor_lead",
    )
    assert final_dec.status == DecisionStatus.ACCEPTED
