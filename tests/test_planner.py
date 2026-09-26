import pytest

from nadi9.domain.enums import ReviewPriority
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Provenance
from nadi9.graph.planner import EpisodePlanner
from nadi9.graph.workflow import build_nadi9_workflow
from nadi9.providers.mock import MockLLMProvider


def make_line(sub_id: str, text: str, context: dict | None = None) -> EpisodeLine:
    return EpisodeLine(
        subtitle_id=sub_id,
        source_text=text,
        speaker="Speaker A",
        scene_id="SCENE-01",
        start_time=0.0,
        end_time=2.0,
        context=context or {},
    )


def make_evidence(ev_id: str, content: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=ev_id,
        source_id="SRC-1",
        evidence_type="dictionary_a",
        content=content,
        provenance=Provenance(original_file="d.json", extraction_method="manual"),
    )


def test_planner_low_risk_subtitle():
    planner = EpisodePlanner()
    line = make_line("SUB-LOW", "Hello world.")
    evidence = [make_evidence("E-1", "hello world greetings")]

    plan = planner.plan_episode("EP-01", [line], evidence)
    assert len(plan.priorities) == 1
    p = plan.priorities[0]

    assert p.subtitle_id == "SUB-LOW"
    assert p.priority == ReviewPriority.LOW
    assert p.risk_score < 0.25
    assert p.requires_deep_reasoning is False


def test_planner_relationship_sensitive_subtitle():
    planner = EpisodePlanner()
    line = make_line("SUB-REL", "Greetings to my elder brother.")
    evidence = [make_evidence("E-1", "elder brother formal terms")]

    plan = planner.plan_episode("EP-01", [line], evidence)
    p = plan.priorities[0]

    assert "relationship_social_hierarchy" in p.risk_reasons
    assert "relationship_terminology_check" in p.required_checks
    assert p.risk_score >= 0.35


def test_planner_tone_sensitive_subtitle():
    planner = EpisodePlanner()
    line = make_line("SUB-TONE", "Formal polite register required.")
    evidence = [make_evidence("E-1", "formal polite terms")]

    plan = planner.plan_episode("EP-01", [line], evidence)
    p = plan.priorities[0]

    assert "tone_register_sensitivity" in p.risk_reasons
    assert "tone_consistency_check" in p.required_checks


def test_planner_name_containing_subtitle():
    planner = EpisodePlanner()
    line = make_line("SUB-NAME", "Welcome Vikram to the assembly.")
    evidence = [make_evidence("E-1", "Vikram assembly")]

    plan = planner.plan_episode("EP-01", [line], evidence)
    p = plan.priorities[0]

    assert "proper_names" in p.risk_reasons
    assert "proper_name_preservation_check" in p.required_checks


def test_planner_number_containing_subtitle():
    planner = EpisodePlanner()
    line = make_line("SUB-NUM", "There are 5 coins in the box.")
    evidence = [make_evidence("E-1", "coins box")]

    plan = planner.plan_episode("EP-01", [line], evidence)
    p = plan.priorities[0]

    assert "numbers_quantities" in p.risk_reasons
    assert "number_preservation_check" in p.required_checks


def test_planner_rare_ambiguous_terminology():
    planner = EpisodePlanner()
    line = make_line("SUB-RARE", "An ancient archaic idiom was spoken.")
    evidence = [make_evidence("E-1", "ancient archaic idiom")]

    plan = planner.plan_episode("EP-01", [line], evidence)
    p = plan.priorities[0]

    assert "cultural_rare_terminology" in p.risk_reasons
    assert "cultural_context_check" in p.required_checks


def test_planner_multiple_risk_factors():
    planner = EpisodePlanner()
    line = make_line("SUB-MULTI", "Formal greeting to elder brother Vikram with 3 gifts.", {"context": "archaic ritual"})
    evidence = []  # Missing evidence to trigger evidence availability risk

    plan = planner.plan_episode("EP-01", [line], evidence)
    p = plan.priorities[0]

    assert p.priority in (ReviewPriority.HIGH, ReviewPriority.CRITICAL)
    assert p.risk_score >= 0.70
    assert p.requires_deep_reasoning is True
    assert "relationship_social_hierarchy" in p.risk_reasons
    assert "proper_names" in p.risk_reasons
    assert "numbers_quantities" in p.risk_reasons
    assert "insufficient_existing_evidence" in p.risk_reasons


def test_planner_priority_ordering():
    planner = EpisodePlanner()
    line_low = make_line("SUB-LOW", "Hello.")
    line_high = make_line("SUB-HIGH", "Formal greeting to elder brother Vikram with 5 gifts.")
    evidence = []

    plan = planner.plan_episode("EP-01", [line_low, line_high], evidence)

    # Highest risk line should come first in priority list
    assert plan.priorities[0].subtitle_id == "SUB-HIGH"
    assert plan.priorities[1].subtitle_id == "SUB-LOW"


def test_planner_deterministic_repeated_planning():
    planner = EpisodePlanner()
    line = make_line("SUB-DET", "Formal greeting to elder brother.")
    evidence = [make_evidence("E-1", "greeting elder brother")]

    plan1 = planner.plan_episode("EP-01", [line], evidence)
    plan2 = planner.plan_episode("EP-01", [line], evidence)

    assert plan1.priorities[0].risk_score == plan2.priorities[0].risk_score
    assert plan1.priorities[0].risk_reasons == plan2.priorities[0].risk_reasons
    assert plan1.priorities[0].required_checks == plan2.priorities[0].required_checks


def test_planning_node_integration_in_workflow():
    provider = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-PLAN-01",
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
    line = make_line("SUB-WF-01", "Formal greeting to elder brother.")
    evidence = [make_evidence("E-001", "Formal greeting to elder brother")]

    initial_state = {
        "run_id": "run-plan-01",
        "episode_id": "ep-plan-01",
        "episode_lines": [line],
        "evidence": evidence,
        "hypotheses": [],
        "conflicts": [],
        "review_items": [],
        "human_review_actions": [],
        "subtitle_decisions": [],
        "changed_evidence_ids": [],
        "affected_subtitle_ids": [],
        "budget": {
            "model_calls_used": 0,
            "tool_calls_used": 0,
            "max_model_calls": 10,
            "max_tool_calls": 10,
        },
        "current_step": "initialized",
        "errors": [],
    }

    final_state = workflow.run(initial_state)

    assert final_state["plan"] is not None
    assert final_state["plan"].episode_id == "ep-plan-01"
    assert len(final_state["plan"].priorities) == 1
    assert final_state["plan"].priorities[0].subtitle_id == "SUB-WF-01"
    assert "relationship_social_hierarchy" in final_state["plan"].priorities[0].risk_reasons
