import pytest

from nadi9.domain.enums import ConfidenceLevel, DecisionStatus, VerificationStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Hypothesis, Provenance
from nadi9.graph.decision import SubtitleDecisionBuilder
from nadi9.graph.state import create_initial_state
from nadi9.graph.verification import HypothesisVerifier
from nadi9.graph.workflow import Nadi9Workflow
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


def test_proper_name_preserved_pass():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Hello Alice and Bob", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Hello Alice and Bob", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.PASSED
    assert res.checks.get("proper_name_preservation") is True


def test_proper_name_changed_fail():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Hello Alice and Bob", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Hello Charlie and Bob", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.FAILED
    assert res.checks.get("proper_name_preservation") is False
    assert any("Proper name preservation" in f for f in res.failures)


def test_integer_preserved_pass():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Give me 25 rupees.", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Give me 25 rupees.", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.PASSED
    assert res.checks.get("number_preservation") is True


def test_number_changed_fail():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Give me 25 rupees.", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Give me 50 rupees.", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.FAILED
    assert res.checks.get("number_preservation") is False
    assert any("Number preservation" in f for f in res.failures)


def test_decimal_preserved_pass():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Rate is 10.5 percent", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Rate is 10.5 percent", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.PASSED
    assert res.checks.get("number_preservation") is True


def test_currency_amount_changed_fail():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Cost is $19.99 today", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Cost is $29.99 today", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.FAILED
    assert res.checks.get("number_preservation") is False


def test_timing_valid_pass():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Short text", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Short text", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.PASSED
    assert res.checks.get("subtitle_timing") is True


def test_reading_speed_too_high_fail():
    verifier = HypothesisVerifier(max_cps=10.0)
    line = EpisodeLine(subtitle_id="S-1", source_text="Quick text", start_time=0.0, end_time=1.0)
    long_statement = "This is an extraordinarily verbose subtitle statement that exceeds reading speed."
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement=long_statement, supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.FAILED
    assert res.checks.get("subtitle_timing") is False
    assert any("Reading speed" in f for f in res.failures)


def test_missing_timing_data_skipped_pass():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="No duration", start_time=0.0, end_time=0.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="No duration", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.checks.get("subtitle_timing") is True


def test_multiple_verification_failures_reported():
    verifier = HypothesisVerifier()
    line = EpisodeLine(subtitle_id="S-1", source_text="Alice has 5 apples", start_time=0.0, end_time=2.0)
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Charlie has 10 apples", supporting_evidence=["E-1"])
    ev = make_evidence("E-1")

    res = verifier.verify(hyp, [ev], episode_line=line)
    assert res.status == VerificationStatus.FAILED
    assert len(res.failures) >= 2


def test_verification_failure_triggers_precise_human_review_question():
    builder = SubtitleDecisionBuilder()
    line = EpisodeLine(subtitle_id="S-1", source_text="Alice has 5 apples", start_time=0.0, end_time=2.0)
    ev = make_evidence("E-1")
    hyp = Hypothesis(hypothesis_id="H-1", category="g", statement="Charlie has 5 apples", supporting_evidence=["E-1"])

    verifier = HypothesisVerifier()
    v_res = verifier.verify(hyp, [ev], episode_line=line)

    decision = builder.build(
        episode=line,
        nadi9_text="Charlie has 5 apples",
        evidence=[ev],
        hypotheses=[hyp],
        verification=v_res,
        conflicts=[],
        confidence=ConfidenceLevel.LOW,
        confidence_reason="Verification failed",
    )

    assert decision.human_review_required is True
    assert decision.review_question is not None
    assert "verify the proper name" in decision.review_question
