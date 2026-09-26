from nadi9.budget import BudgetManager
from nadi9.domain.enums import DecisionStatus, VerificationStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Provenance
from nadi9.graph.state import create_initial_state
from nadi9.graph.workflow import Nadi9Workflow, build_nadi9_workflow
from nadi9.providers import BudgetedLLMProvider, MockLLMProvider


def make_evidence(
    evidence_id: str,
    content: str,
    conflicts_with: list[str] | None = None,
) -> EvidenceRecord:
    metadata = {}
    if conflicts_with:
        metadata["conflicts_with"] = conflicts_with

    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-001",
        evidence_type="dictionary_a",
        content=content,
        metadata=metadata,
        provenance=Provenance(
            original_file="test.json",
            location="entry-1",
            extraction_method="fixture",
        ),
    )


def make_episode() -> EpisodeLine:
    return EpisodeLine(
        subtitle_id="SUB-001",
        source_text="formal greeting",
        speaker="Speaker A",
        scene_id="SCENE-001",
        start_time=0.0,
        end_time=2.0,
    )


def test_workflow_happy_path_finalizes():
    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar is the formal greeting.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            }
        ]
    )

    workflow = build_nadi9_workflow(provider)
    initial_state = create_initial_state("run-1", "ep-1", episode_lines=[make_episode()])
    initial_state["evidence"] = [
        make_evidence("E-001", "formal greeting used by elders")
    ]

    final_state = workflow.run(initial_state)

    assert final_state["current_step"] == "finalize"
    assert len(final_state["hypotheses"]) == 1
    assert len(final_state["subtitle_decisions"]) == 1

    decision = final_state["subtitle_decisions"][0]
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.nadi9_text == "Namaskar is the formal greeting."
    assert decision.human_review_required is False
    assert decision.verification.status == VerificationStatus.PASSED
    assert len(final_state["errors"]) == 0


def test_workflow_routes_to_human_review_on_missing_evidence():
    provider = MockLLMProvider(responses=[])
    workflow = build_nadi9_workflow(provider)

    initial_state = create_initial_state("run-1", "ep-1", episode_lines=[make_episode()])
    initial_state["evidence"] = []

    final_state = workflow.run(initial_state)

    assert final_state["current_step"] == "human_review"
    assert len(final_state["errors"]) > 0
    assert "No relevant evidence" in final_state["errors"][0]


def test_workflow_routes_to_human_review_on_verification_failure():
    # Hypothesis cites non-existent evidence ID E-999
    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-999"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )
    workflow = build_nadi9_workflow(provider)

    initial_state = create_initial_state("run-1", "ep-1", episode_lines=[make_episode()])
    initial_state["evidence"] = [
        make_evidence("E-001", "formal greeting used by elders")
    ]

    final_state = workflow.run(initial_state)

    assert final_state["current_step"] == "human_review"
    assert len(final_state["errors"]) > 0


def test_workflow_routes_to_human_review_on_conflict():
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
    workflow = build_nadi9_workflow(provider)

    initial_state = create_initial_state("run-1", "ep-1", episode_lines=[make_episode()])
    initial_state["evidence"] = [
        make_evidence(
            "E-001", "formal greeting", conflicts_with=["E-002"]
        ),
        make_evidence("E-002", "informal greeting"),
    ]

    final_state = workflow.run(initial_state)

    assert final_state["current_step"] == "human_review"
    assert len(final_state["conflicts"]) == 1
    assert len(final_state["subtitle_decisions"]) == 1

    decision = final_state["subtitle_decisions"][0]
    assert decision.status == DecisionStatus.REVIEW_REQUIRED
    assert decision.human_review_required is True
    assert len(final_state["review_items"]) == 1


def test_workflow_with_budgeted_provider_exhaustion():
    mock = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )

    budget_mgr = BudgetManager(max_model_calls=1)
    budget_mgr.consume_model_call()  # Exhaust the budget before invocation

    budgeted_provider = BudgetedLLMProvider(mock, budget_mgr)

    workflow = Nadi9Workflow(budgeted_provider)

    initial_state = create_initial_state("run-1", "ep-1", episode_lines=[make_episode()])
    initial_state["evidence"] = [
        make_evidence("E-001", "formal greeting used by elders")
    ]

    final_state = workflow.run(initial_state)

    assert final_state["current_step"] == "human_review"
    assert len(final_state["errors"]) > 0
    assert "budget" in final_state["errors"][0].lower() or "exceeded" in final_state["errors"][0].lower()


def test_workflow_state_integrity():
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
    workflow = build_nadi9_workflow(provider)

    initial_state = create_initial_state("run-42", "ep-99", episode_lines=[make_episode()])
    initial_state["evidence"] = [
        make_evidence("E-001", "formal greeting used by elders")
    ]

    final_state = workflow.run(initial_state)

    assert final_state["run_id"] == "run-42"
    assert final_state["episode_id"] == "ep-99"
    assert final_state["budget"]["max_model_calls"] == 25
    assert len(final_state["subtitle_decisions"]) == 1
