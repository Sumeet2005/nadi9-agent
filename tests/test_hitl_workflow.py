import pytest

from nadi9.budget import BudgetManager
from nadi9.domain.enums import DecisionStatus, VerificationStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Provenance
from nadi9.graph.state import create_initial_state
from nadi9.graph.workflow import Nadi9Workflow, build_nadi9_workflow
from nadi9.processor import EpisodeProcessor
from nadi9.providers import MockLLMProvider
from nadi9.storage import create_db_engine, create_session_factory, init_db


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
        subtitle_id="SUB-HITL-01",
        source_text="formal greeting",
        speaker="Speaker A",
        scene_id="SCENE-01",
        start_time=0.0,
        end_time=2.0,
    )


@pytest.fixture
def mock_provider():
    return MockLLMProvider(
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


def test_workflow_interrupts_for_human_review(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-1", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    interrupted_state = workflow.run(state, thread_id="thread-001")

    assert interrupted_state["current_step"] == "human_review"
    assert len(interrupted_state["subtitle_decisions"]) == 1
    assert interrupted_state["subtitle_decisions"][0].human_review_required is True
    assert "__interrupt__" in interrupted_state


def test_checkpoint_contains_expected_state(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-2", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-002")

    checkpoint_snapshot = workflow.get_state("thread-002")
    assert checkpoint_snapshot is not None
    assert checkpoint_snapshot.next == ("route_human_review",)
    assert len(checkpoint_snapshot.values["subtitle_decisions"]) == 1


def test_approve_resumes_workflow(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-3", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-003")

    action = {"action_type": "approve", "actor": "reviewer_alice"}
    final_state = workflow.resume("thread-003", action)

    assert final_state["current_step"] == "finalize"
    assert len(final_state["subtitle_decisions"]) == 1
    decision = final_state["subtitle_decisions"][0]
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.human_review_required is False
    assert "Approved by human reviewer" in decision.confidence_reason


def test_reject_resumes_workflow(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-4", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-004")

    action = {
        "action_type": "reject",
        "reason": "Translation is grammatically inappropriate for scene context.",
        "actor": "reviewer_bob",
    }
    final_state = workflow.resume("thread-004", action)

    assert final_state["current_step"] == "finalize"
    assert len(final_state["subtitle_decisions"]) == 1
    decision = final_state["subtitle_decisions"][0]
    assert decision.status == DecisionStatus.REJECTED
    assert decision.human_review_required is False
    assert "Rejected by human reviewer" in decision.confidence_reason


def test_correct_resumes_workflow(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-5", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-005")

    action = {
        "action_type": "correct",
        "text": "Pranam, Guruji!",
        "reason": "Respectful register correction.",
        "actor": "reviewer_carol",
    }
    final_state = workflow.resume("thread-005", action)

    assert final_state["current_step"] == "finalize"
    assert len(final_state["subtitle_decisions"]) == 1
    decision = final_state["subtitle_decisions"][0]
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.nadi9_text == "Pranam, Guruji!"
    assert decision.human_review_required is False


def test_resume_does_not_regenerate_hypothesis_and_no_extra_llm_calls(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-6", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-006")
    calls_before_resume = mock_provider.call_count

    action = {"action_type": "approve", "actor": "reviewer_david"}
    final_state = workflow.resume("thread-006", action)
    calls_after_resume = mock_provider.call_count

    assert calls_after_resume == calls_before_resume
    assert len(final_state["hypotheses"]) == 1


def test_review_action_is_audited(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-7", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-007")

    action = {
        "action_type": "approve",
        "actor": "reviewer_audit",
        "reason": "Audit verification passed.",
    }
    final_state = workflow.resume("thread-007", action)

    assert len(final_state["human_review_actions"]) == 1
    rec = final_state["human_review_actions"][0]
    assert rec.action_type == "approve"
    assert rec.actor == "reviewer_audit"
    assert rec.subtitle_id == "SUB-HITL-01"


def test_original_ai_text_is_preserved_after_correction(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-8", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-008")

    action = {
        "action_type": "correct",
        "text": "Corrected greeting text",
        "actor": "reviewer_eve",
    }
    final_state = workflow.resume("thread-008", action)

    rec = final_state["human_review_actions"][0]
    assert rec.original_nadi9_text == "Namaskar."
    assert rec.corrected_text == "Corrected greeting text"


def test_reject_requires_reason(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-9", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-009")

    action = {"action_type": "reject", "reason": "   ", "actor": "reviewer_frank"}
    with pytest.raises(ValueError, match="Rejection reason cannot be empty"):
        workflow.resume("thread-009", action)


def test_correct_requires_text(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    state = create_initial_state("run-hitl-10", "ep-1", episode_lines=[make_episode()])
    state["evidence"] = make_conflicting_evidence()

    workflow.run(state, thread_id="thread-010")

    action = {"action_type": "correct", "text": "", "actor": "reviewer_grace"}
    with pytest.raises(ValueError, match="Corrected text cannot be empty"):
        workflow.resume("thread-010", action)


def test_unknown_thread_fails_cleanly(mock_provider):
    workflow = build_nadi9_workflow(mock_provider)
    action = {"action_type": "approve", "actor": "reviewer_unknown"}

    with pytest.raises(Exception):
        workflow.resume("non_existent_thread_999", action)


def test_episode_processor_resume_human_review(tmp_path, mock_provider):
    db_file = tmp_path / "test_hitl_proc.db"
    engine = create_db_engine(f"sqlite:///{db_file}")
    init_db(engine)
    session_factory = create_session_factory(engine)
    session = session_factory()

    processor = EpisodeProcessor(provider=mock_provider, db_session=session)

    ep_lines = [make_episode()]
    evidence = make_conflicting_evidence()

    report = processor.process_records(
        episode_lines=ep_lines,
        evidence=evidence,
        run_id="run-proc-hitl-1",
    )

    assert report.human_review_count == 1

    final_decision = processor.resume_human_review(
        run_id="run-proc-hitl-1",
        subtitle_id="SUB-HITL-01",
        action_type="approve",
        actor="lead_editor",
    )

    assert final_decision.status == DecisionStatus.ACCEPTED
    assert final_decision.human_review_required is False
    session.close()
