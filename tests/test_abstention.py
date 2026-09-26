import pytest

from nadi9.domain.enums import ConfidenceLevel, DecisionStatus, VerificationStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Hypothesis, Provenance, VerificationResult, Conflict
from nadi9.graph.decision import SubtitleDecisionBuilder
from nadi9.graph.state import create_initial_state
from nadi9.graph.workflow import Nadi9Workflow
from nadi9.processor import EpisodeProcessor
from nadi9.providers.mock import MockLLMProvider


def make_evidence(evidence_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-1",
        evidence_type="approved_example",
        content="formal greeting sample content",
        provenance=Provenance(
            original_file="sample.json",
            location="1",
            extraction_method="mock",
        ),
    )


def test_empty_evidence_triggers_explicit_abstention():
    line = EpisodeLine(subtitle_id="L-NOEVID", source_text="Unseen ancient phrase", start_time=0.0, end_time=1.0)
    provider = MockLLMProvider(responses=[])

    line_state = create_initial_state("run-1", "ep-1", [line])
    line_state["evidence"] = []
    line_state["episode"] = line
    line_state["latest_verification"] = VerificationResult(
        status=VerificationStatus.FAILED,
        checks={"has_supporting_evidence": False},
        failures=["Hypothesis does not cite supporting evidence."],
    )

    workflow = Nadi9Workflow(provider=provider)
    res = workflow._node_generate_decision(line_state)

    decision = res["subtitle_decisions"][0]
    assert decision.nadi9_text == "Translation unavailable: insufficient supporting evidence."
    assert decision.human_review_required is True
    assert decision.confidence == ConfidenceLevel.LOW
    assert decision.abstained is True
    assert "no supporting evidence was found" in decision.review_question


def test_conflicting_evidence_triggers_explicit_abstention_and_precise_question():
    line = EpisodeLine(subtitle_id="L-CONF", source_text="Ambiguous honorific", start_time=0.0, end_time=1.0)
    ev1 = make_evidence("E-1")
    ev2 = make_evidence("E-2")
    conflict = Conflict(conflict_id="CONF-101", description="Sources disagree on formal vs informal register", evidence_ids=["E-1", "E-2"])

    builder = SubtitleDecisionBuilder()
    verification = VerificationResult(
        status=VerificationStatus.PASSED,
        checks={"supporting_evidence_exists": True},
        failures=[],
        evidence_checked=["E-1", "E-2"],
    )

    decision = builder.build(
        episode=line,
        nadi9_text="Translation unavailable: insufficient supporting evidence.",
        evidence=[ev1, ev2],
        hypotheses=[],
        verification=verification,
        conflicts=[conflict],
        confidence=ConfidenceLevel.LOW,
        confidence_reason="Conflicting evidence sources.",
    )

    assert decision.nadi9_text == "Translation unavailable: insufficient supporting evidence."
    assert decision.human_review_required is True
    assert decision.status == DecisionStatus.REVIEW_REQUIRED
    assert decision.abstained is True
    assert "Please resolve the conflicting evidence" in decision.review_question
    assert "CONF-101" in decision.review_question


def test_verification_failure_triggers_precise_questions():
    builder = SubtitleDecisionBuilder()
    line = EpisodeLine(subtitle_id="L-VERIF", source_text="King Arthur had 12 knights", start_time=0.0, end_time=1.0)

    # Name preservation failure
    name_verif = VerificationResult(
        status=VerificationStatus.FAILED,
        checks={"proper_name_preservation": False},
        failures=["Proper name check failed"],
    )
    q_name = builder._build_review_question(verification=name_verif, conflicts=[])
    assert "verify the proper name" in q_name

    # Number preservation failure
    num_verif = VerificationResult(
        status=VerificationStatus.FAILED,
        checks={"number_preservation": False},
        failures=["Number quantity check failed"],
    )
    q_num = builder._build_review_question(verification=num_verif, conflicts=[])
    assert "verify the numeric value" in q_num

    # General verification failure
    gen_verif = VerificationResult(
        status=VerificationStatus.FAILED,
        checks={},
        failures=["Grammar rule check failed"],
    )
    q_gen = builder._build_review_question(verification=gen_verif, conflicts=[])
    assert "independent verification failed: Grammar rule check failed" in q_gen


def test_successful_grounded_translation_is_not_abstained():
    line = EpisodeLine(subtitle_id="L-OK", source_text="Hello friend", start_time=0.0, end_time=1.0)
    ev = make_evidence("E-1")
    hyp = Hypothesis(hypothesis_id="H-1", category="grammar", statement="Hello friend", confidence=ConfidenceLevel.HIGH)

    builder = SubtitleDecisionBuilder()
    verif = VerificationResult(
        status=VerificationStatus.PASSED,
        checks={"supporting_evidence_exists": True},
        failures=[],
        evidence_checked=["E-1"],
    )

    decision = builder.build(
        episode=line,
        nadi9_text="Hello friend",
        evidence=[ev],
        hypotheses=[hyp],
        verification=verif,
        conflicts=[],
        confidence=ConfidenceLevel.HIGH,
        confidence_reason="Verified decision",
    )

    assert decision.nadi9_text == "Hello friend"
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.human_review_required is False
    assert decision.abstained is False
    assert decision.review_question is None


def test_abstention_traceable_in_decision_data():
    line = EpisodeLine(subtitle_id="L-TRACE", source_text="Rare idiom", start_time=0.0, end_time=1.0)
    builder = SubtitleDecisionBuilder()
    verif = VerificationResult(
        status=VerificationStatus.FAILED,
        checks={"has_supporting_evidence": False},
        failures=["No evidence"],
    )

    decision = builder.build(
        episode=line,
        nadi9_text="Translation unavailable: insufficient supporting evidence.",
        evidence=[],
        hypotheses=[],
        verification=verif,
        conflicts=[],
        confidence=ConfidenceLevel.LOW,
        confidence_reason="No evidence grounded hypothesis.",
    )

    dumped = decision.model_dump(mode="json")
    assert dumped["subtitle_id"] == "L-TRACE"
    assert dumped["source_text"] == "Rare idiom"
    assert dumped["nadi9_text"] == "Translation unavailable: insufficient supporting evidence."
    assert dumped["abstained"] is True
    assert dumped["verification"]["status"] == "failed"
