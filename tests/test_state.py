import pytest

from nadi9.graph.state import create_initial_state


def test_initial_state_has_expected_structure():
    state = create_initial_state(
        run_id="RUN-001",
        episode_id="EP-001",
    )

    assert state["run_id"] == "RUN-001"
    assert state["episode_id"] == "EP-001"

    assert state["evidence"] == []
    assert state["hypotheses"] == []
    assert state["conflicts"] == []
    assert state["review_items"] == []
    assert state["subtitle_decisions"] == []

    assert state["changed_evidence_ids"] == []
    assert state["affected_subtitle_ids"] == []
    assert state["errors"] == []


def test_initial_budget_matches_assignment_limits():
    state = create_initial_state(
        run_id="RUN-001",
        episode_id="EP-001",
    )

    assert state["budget"]["model_calls_used"] == 0
    assert state["budget"]["tool_calls_used"] == 0
    assert state["budget"]["max_model_calls"] == 25
    assert state["budget"]["max_tool_calls"] == 50


def test_custom_budget_can_be_configured():
    state = create_initial_state(
        run_id="RUN-002",
        episode_id="EP-002",
        max_model_calls=10,
        max_tool_calls=20,
    )

    assert state["budget"]["max_model_calls"] == 10
    assert state["budget"]["max_tool_calls"] == 20


def test_invalid_model_budget_is_rejected():
    with pytest.raises(ValueError, match="max_model_calls"):
        create_initial_state(
            run_id="RUN-003",
            episode_id="EP-003",
            max_model_calls=0,
        )


def test_invalid_tool_budget_is_rejected():
    with pytest.raises(ValueError, match="max_tool_calls"):
        create_initial_state(
            run_id="RUN-004",
            episode_id="EP-004",
            max_tool_calls=0,
        )