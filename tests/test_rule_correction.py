from typing import Any
import pytest

from nadi9.domain.enums import ConfidenceLevel, DecisionStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, LearnedRule, SubtitleDecision, Provenance
from nadi9.graph.planner import EpisodePlanner
from nadi9.graph.state import create_initial_state, merge_learned_rules
from nadi9.processor import EpisodeProcessor
from nadi9.providers.mock import MockLLMProvider


def make_evidence(evidence_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-1",
        evidence_type="approved_example",
        content="formal greeting test content",
        provenance=Provenance(
            original_file="test.json",
            location="1",
            extraction_method="mock",
        ),
    )


def test_rule_correction_supersedes_and_preserves_history():
    rule_v1 = LearnedRule(
        rule_id="RULE-100",
        category="honorifics",
        statement="Translate greeting as 'Hello Sir'",
        supporting_evidence=["E-1"],
        affected_subtitle_ids=["L002", "L003"],
    )
    rule_v2 = LearnedRule(
        rule_id="RULE-101",
        category="honorifics",
        statement="Translate greeting as 'Respected Elder'",
        supporting_evidence=["E-1", "E-2"],
        affected_subtitle_ids=["L002", "L003"],
        supersedes="RULE-100",
    )

    merged = merge_learned_rules([rule_v1], [rule_v2])

    assert len(merged) == 2
    assert merged[0].rule_id == "RULE-100"
    assert merged[1].rule_id == "RULE-101"
    assert merged[1].supersedes == "RULE-100"
    assert merged[1].affected_subtitle_ids == ["L002", "L003"]


test_affected_subtitle_ids_identified_and_unaffected_preserved = test_rule_correction_supersedes_and_preserves_history


def test_selective_invalidation_and_targeted_replanning():
    planner = EpisodePlanner()
    lines = [
        EpisodeLine(subtitle_id="L001", source_text="Hello", start_time=0.0, end_time=1.0),
        EpisodeLine(subtitle_id="L002", source_text="Good morning sir", start_time=1.0, end_time=2.0),
        EpisodeLine(subtitle_id="L003", source_text="Respectful farewell", start_time=2.0, end_time=3.0),
        EpisodeLine(subtitle_id="L004", source_text="Bye", start_time=3.0, end_time=4.0),
    ]

    targeted_plan = planner.plan_subset(
        episode_id="EP-1",
        episode_lines=lines,
        target_subtitle_ids=["L002", "L003"],
    )

    assert targeted_plan.status == "targeted_replan"
    assert len(targeted_plan.priorities) == 2
    planned_ids = {p.subtitle_id for p in targeted_plan.priorities}
    assert planned_ids == {"L002", "L003"}


def test_apply_rule_correction_selective_reprocessing_end_to_end():
    lines = [
        EpisodeLine(subtitle_id="L001", source_text="Hello friend", start_time=0.0, end_time=1.0),
        EpisodeLine(subtitle_id="L002", source_text="Good morning elder", start_time=1.0, end_time=2.0),
        EpisodeLine(subtitle_id="L003", source_text="Farewell sir", start_time=2.0, end_time=3.0),
        EpisodeLine(subtitle_id="L004", source_text="Goodbye", start_time=3.0, end_time=4.0),
    ]
    evidence = [make_evidence("E-1")]

    mock_resps = [
        {"hypothesis_id": "H1", "category": "grammar", "statement": "Hello friend", "confidence": "high"},
        {"hypothesis_id": "H2", "category": "grammar", "statement": "Good morning elder", "confidence": "medium"},
        {"hypothesis_id": "H3", "category": "grammar", "statement": "Farewell sir", "confidence": "medium"},
        {"hypothesis_id": "H4", "category": "grammar", "statement": "Goodbye", "confidence": "high"},
    ]

    # Provider with initial runs + reprocessed runs
    reprocess_resps = [
        {"hypothesis_id": "H2-v2", "category": "grammar", "statement": "Respected morning elder", "confidence": "high"},
        {"hypothesis_id": "H3-v2", "category": "grammar", "statement": "Respected farewell sir", "confidence": "high"},
    ]

    provider = MockLLMProvider(responses=mock_resps + reprocess_resps)
    processor = EpisodeProcessor(provider=provider)

    initial_report = processor.process_records(
        episode_lines=lines,
        evidence=evidence,
        run_id="run-100",
        episode_id="ep-100",
    )

    initial_state = create_initial_state("run-100", "ep-100", lines)
    initial_state["subtitle_decisions"] = initial_report.decisions
    initial_state["evidence"] = evidence

    # Record initial L001 decision object reference to verify it remains unchanged
    unaffected_l001_initial = [d for d in initial_state["subtitle_decisions"] if d.subtitle_id == "L001"][0]

    rule_v1 = LearnedRule(
        rule_id="R-1",
        category="honorifics",
        statement="Use Elder",
        supporting_evidence=["E-1"],
        affected_subtitle_ids=["L002", "L003"],
    )
    rule_v2 = LearnedRule(
        rule_id="R-2",
        category="honorifics",
        statement="Use Respected",
        supporting_evidence=["E-1"],
        affected_subtitle_ids=["L002", "L003"],
        supersedes="R-1",
    )

    updated_state = processor.apply_rule_correction(
        overall_state=initial_state,
        new_rule=rule_v2,
        episode_lines=lines,
        evidence=evidence,
    )

    decisions_by_id = {d.subtitle_id: d for d in updated_state["subtitle_decisions"]}

    assert len(decisions_by_id) == 4
    # Unaffected L001 and L004 preserved
    assert decisions_by_id["L001"] == unaffected_l001_initial
    assert decisions_by_id["L004"].nadi9_text == "Goodbye"

    # Affected L002 and L003 reprocessed
    assert decisions_by_id["L002"].nadi9_text == "Respected morning elder"
    assert decisions_by_id["L003"].nadi9_text == "Respected farewell sir"

    # Learned rules updated and superseded history preserved
    assert len(updated_state["learned_rules"]) == 1
    assert updated_state["learned_rules"][0].rule_id == "R-2"
    assert updated_state["learned_rules"][0].supersedes == "R-1"


def test_rule_affecting_zero_subtitles_handled_safely():
    provider = MockLLMProvider(responses=[])
    processor = EpisodeProcessor(provider=provider)

    rule = LearnedRule(
        rule_id="R-EMPTY",
        category="test",
        statement="No effect",
        affected_subtitle_ids=[],
    )

    state = create_initial_state("run-1", "ep-1")
    updated = processor.apply_rule_correction(
        overall_state=state,
        new_rule=rule,
        episode_lines=[],
        evidence=[],
    )
    assert updated == state


def test_checkpoint_state_remains_valid_after_correction():
    mock_resp = {"hypothesis_id": "H1", "category": "grammar", "statement": "Test line", "confidence": "high"}
    provider = MockLLMProvider(responses=[mock_resp, mock_resp])
    processor = EpisodeProcessor(provider=provider)

    lines = [EpisodeLine(subtitle_id="L-CHK", source_text="Checkpoint line", start_time=0.0, end_time=1.0)]
    state = create_initial_state("run-chk", "ep-chk", lines)

    rule = LearnedRule(rule_id="R-CHK", category="test", statement="Check rule", affected_subtitle_ids=["L-CHK"])
    updated = processor.apply_rule_correction(
        overall_state=state,
        new_rule=rule,
        episode_lines=lines,
        evidence=[],
    )

    assert "subtitle_decisions" in updated
    assert len(updated["subtitle_decisions"]) == 1
    assert updated["subtitle_decisions"][0].subtitle_id == "L-CHK"
