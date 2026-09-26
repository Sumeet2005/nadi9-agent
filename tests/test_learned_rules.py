from datetime import UTC, datetime
from typing import Any
import pytest
from pydantic import ValidationError

from nadi9.domain.enums import ConfidenceLevel, DecisionStatus
from nadi9.domain.models import LearnedRule, EpisodeLine, SubtitleDecision
from nadi9.graph.state import AgentState, create_initial_state, merge_learned_rules
from nadi9.graph.workflow import Nadi9Workflow
from nadi9.providers.mock import MockLLMProvider


def test_learned_rule_creation_valid():
    rule = LearnedRule(
        rule_id="RULE-001",
        category="grammar",
        statement="Use formal honorific prefix before elder names",
        supporting_evidence=["E-001", "E-002"],
        affected_subtitle_ids=["SUB-101", "SUB-102"],
        confidence=ConfidenceLevel.HIGH,
        supersedes=None,
    )
    assert rule.rule_id == "RULE-001"
    assert rule.category == "grammar"
    assert len(rule.supporting_evidence) == 2
    assert len(rule.affected_subtitle_ids) == 2
    assert rule.confidence == ConfidenceLevel.HIGH
    assert rule.supersedes is None
    assert isinstance(rule.created_at, datetime)


def test_learned_rule_field_validation():
    with pytest.raises(ValidationError):
        # Empty statement is disallowed by min_length=1
        LearnedRule(
            rule_id="RULE-001",
            category="grammar",
            statement="",
        )

    with pytest.raises(ValidationError):
        # Missing required fields rule_id and category
        LearnedRule(statement="Valid statement")


def test_learned_rule_default_list_independence():
    rule1 = LearnedRule(rule_id="R-1", category="c1", statement="s1")
    rule2 = LearnedRule(rule_id="R-2", category="c2", statement="s2")

    rule1.supporting_evidence.append("E-1")
    assert "E-1" not in rule2.supporting_evidence


def test_agent_state_multiple_coexisting_learned_rules():
    rule1 = LearnedRule(rule_id="R-1", category="cat1", statement="stmt1")
    rule2 = LearnedRule(rule_id="R-2", category="cat2", statement="stmt2")

    state = create_initial_state("run-1", "ep-1")
    state["learned_rules"] = merge_learned_rules(state["learned_rules"], [rule1, rule2])

    assert len(state["learned_rules"]) == 2
    assert state["learned_rules"][0].rule_id == "R-1"
    assert state["learned_rules"][1].rule_id == "R-2"


def test_rule_ids_affected_subtitles_supporting_evidence_preserved():
    rule = LearnedRule(
        rule_id="RULE-PERSIST-100",
        category="honorifics",
        statement="Always preserve regional greeting particle",
        supporting_evidence=["EVID-10", "EVID-11"],
        affected_subtitle_ids=["SUB-001", "SUB-002", "SUB-003"],
        confidence=ConfidenceLevel.MEDIUM,
        supersedes="RULE-PERSIST-099",
    )

    assert rule.rule_id == "RULE-PERSIST-100"
    assert rule.affected_subtitle_ids == ["SUB-001", "SUB-002", "SUB-003"]
    assert rule.supporting_evidence == ["EVID-10", "EVID-11"]
    assert rule.supersedes == "RULE-PERSIST-099"


def test_new_rule_supersedes_existing_rule_preserving_audit_history():
    rule_v1 = LearnedRule(
        rule_id="RULE-V1",
        category="dialect",
        statement="Translate particle as 'indeed'",
        supporting_evidence=["E-1"],
        affected_subtitle_ids=["SUB-1"],
    )
    rule_v2 = LearnedRule(
        rule_id="RULE-V2",
        category="dialect",
        statement="Translate particle as 'certainly' in formal contexts",
        supporting_evidence=["E-1", "E-2"],
        affected_subtitle_ids=["SUB-1", "SUB-2"],
        supersedes="RULE-V1",
    )

    merged = merge_learned_rules([rule_v1], [rule_v2])

    assert len(merged) == 2
    assert merged[0].rule_id == "RULE-V1"
    assert merged[1].rule_id == "RULE-V2"
    assert merged[1].supersedes == "RULE-V1"


def test_learned_rule_serialization_deserialization():
    rule = LearnedRule(
        rule_id="RULE-SER-1",
        category="idiom",
        statement="Literal idiom translation rule",
        supporting_evidence=["E-100"],
        affected_subtitle_ids=["SUB-50"],
        supersedes="RULE-OLD",
    )

    dumped = rule.model_dump(mode="json")
    assert dumped["rule_id"] == "RULE-SER-1"
    assert dumped["category"] == "idiom"
    assert dumped["supersedes"] == "RULE-OLD"

    restored = LearnedRule.model_validate(dumped)
    assert restored.rule_id == rule.rule_id
    assert restored.statement == rule.statement
    assert restored.supersedes == rule.supersedes


def test_existing_agent_state_creation_unbroken():
    state = create_initial_state("run-123", "ep-456")
    assert state["run_id"] == "run-123"
    assert state["learned_rules"] == []
    assert "plan" in state
    assert "budget" in state


def test_graph_state_learned_rule_integration(tmp_path):
    mock_resp = {
        "hypothesis_id": "HYP-999",
        "category": "grammar",
        "statement": "Original candidate subtitle",
        "supporting_evidence": [],
        "counterexamples": [],
        "status": "proposed",
        "confidence": "medium",
    }
    provider = MockLLMProvider(responses=[mock_resp])
    workflow = Nadi9Workflow(provider=provider)

    line = EpisodeLine(
        subtitle_id="SUB-999",
        source_text="Test source text",
        start_time=0.0,
        end_time=2.0,
    )
    state = create_initial_state("run-test", "ep-test", [line])

    # Run workflow until review interrupt
    try:
        res = workflow.run(state)
    except Exception:
        pass

    state_snap = workflow.get_state("run-test:SUB-999")
    assert state_snap.next == ("route_human_review",)

    # Resume with correction action which produces a LearnedRule
    action = {
        "action_type": "correct",
        "actor": "linguist",
        "reason": "Fix dialect spelling",
        "corrected_text": "Corrected subtitle text",
    }
    final_state = workflow.resume("run-test:SUB-999", action)

    assert len(final_state.get("learned_rules", [])) == 1
    rule = final_state["learned_rules"][0]
    assert rule.affected_subtitle_ids == ["SUB-999"]
    assert "Corrected subtitle text" in rule.statement
