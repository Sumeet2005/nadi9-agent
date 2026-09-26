from nadi9.domain.models import (
    Conflict,
    EpisodeLine,
    EvidenceRecord,
    Hypothesis,
    Provenance,
    VerificationResult,
)
from nadi9.domain.enums import (
    ConfidenceLevel,
    DecisionStatus,
    VerificationStatus,
)
from nadi9.graph.decision import (
    SubtitleDecisionBuilder,
    SubtitleDecisionError,
)


def make_episode() -> EpisodeLine:
    return EpisodeLine(
        subtitle_id="SUB-001",
        source_text="Hello.",
        speaker="Speaker 1",
        scene_id="SCENE-001",
        start_time=1.0,
        end_time=3.0,
    )


def make_evidence(
    evidence_id: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-001",
        evidence_type="dictionary_a",
        content="Greeting evidence.",
        provenance=Provenance(
            original_file="test.json",
            location="entry-1",
            extraction_method="fixture",
        ),
    )


def make_hypothesis(
    hypothesis_id: str = "H-001",
    evidence_ids: list[str] | None = None,
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        category="greeting",
        statement="Use the formal greeting.",
        supporting_evidence=evidence_ids or ["E-001"],
        counterexamples=[],
        status="proposed",
        confidence="medium",
    )


def make_verification(
    status: VerificationStatus = VerificationStatus.PASSED,
) -> VerificationResult:
    return VerificationResult(
        status=status,
        checks={
            "supporting_evidence_exists": status
            == VerificationStatus.PASSED,
        },
        failures=(
            []
            if status == VerificationStatus.PASSED
            else ["Evidence verification failed."]
        ),
        evidence_checked=["E-001"],
    )


def test_builder_creates_accepted_decision():
    builder = SubtitleDecisionBuilder()

    decision = builder.build(
        episode=make_episode(),
        nadi9_text="Namaskar.",
        evidence=[make_evidence("E-001")],
        hypotheses=[make_hypothesis()],
        verification=make_verification(),
        confidence=ConfidenceLevel.HIGH,
        confidence_reason="Supported by approved evidence.",
    )

    assert decision.subtitle_id == "SUB-001"
    assert decision.source_text == "Hello."
    assert decision.nadi9_text == "Namaskar."
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.evidence_ids == ["E-001"]
    assert decision.hypothesis_ids == ["H-001"]
    assert decision.confidence == ConfidenceLevel.HIGH
    assert decision.human_review_required is False
    assert decision.review_question is None


def test_builder_requires_review_for_failed_verification():
    builder = SubtitleDecisionBuilder()

    decision = builder.build(
        episode=make_episode(),
        nadi9_text="Namaskar.",
        evidence=[make_evidence("E-001")],
        hypotheses=[make_hypothesis()],
        verification=make_verification(
            VerificationStatus.FAILED,
        ),
        confidence=ConfidenceLevel.LOW,
        confidence_reason="Verification failed.",
    )

    assert decision.status == DecisionStatus.REVIEW_REQUIRED
    assert decision.human_review_required is True
    assert decision.review_question is not None
    assert "verification failed" in decision.review_question.lower() or "verification failures" in decision.review_question.lower()


def test_builder_requires_review_for_conflict():
    builder = SubtitleDecisionBuilder()

    conflict = Conflict(
        conflict_id="CONFLICT-E-001-E-002",
        description="Evidence items disagree.",
        evidence_ids=["E-001", "E-002"],
        resolved=False,
    )

    decision = builder.build(
        episode=make_episode(),
        nadi9_text="Namaskar.",
        evidence=[
            make_evidence("E-001"),
            make_evidence("E-002"),
        ],
        hypotheses=[
            make_hypothesis(
                evidence_ids=["E-001", "E-002"],
            ),
        ],
        verification=make_verification(),
        conflicts=[conflict],
        confidence=ConfidenceLevel.MEDIUM,
        confidence_reason="Evidence requires review.",
    )

    assert decision.status == DecisionStatus.REVIEW_REQUIRED
    assert decision.human_review_required is True
    assert decision.conflicts == [
        "CONFLICT-E-001-E-002"
    ]
    assert decision.review_question is not None
    assert "CONFLICT-E-001-E-002" in decision.review_question


def test_builder_deduplicates_evidence_ids():
    builder = SubtitleDecisionBuilder()

    hypotheses = [
        make_hypothesis(
            hypothesis_id="H-001",
            evidence_ids=["E-001", "E-002"],
        ),
        make_hypothesis(
            hypothesis_id="H-002",
            evidence_ids=["E-002", "E-003"],
        ),
    ]

    decision = builder.build(
        episode=make_episode(),
        nadi9_text="Namaskar.",
        evidence=[
            make_evidence("E-001"),
            make_evidence("E-002"),
            make_evidence("E-003"),
        ],
        hypotheses=hypotheses,
        verification=make_verification(),
        confidence=ConfidenceLevel.HIGH,
        confidence_reason="Multiple supporting hypotheses.",
    )

    assert decision.evidence_ids == [
        "E-001",
        "E-002",
        "E-003",
    ]
    assert decision.hypothesis_ids == [
        "H-001",
        "H-002",
    ]


def test_builder_ignores_unknown_evidence_ids():
    builder = SubtitleDecisionBuilder()

    hypothesis = make_hypothesis(
        evidence_ids=["E-001", "E-999"],
    )

    decision = builder.build(
        episode=make_episode(),
        nadi9_text="Namaskar.",
        evidence=[make_evidence("E-001")],
        hypotheses=[hypothesis],
        verification=make_verification(),
        confidence=ConfidenceLevel.MEDIUM,
        confidence_reason="Partial evidence available.",
    )

    assert decision.evidence_ids == ["E-001"]


def test_builder_rejects_empty_nadi9_text():
    builder = SubtitleDecisionBuilder()

    try:
        builder.build(
            episode=make_episode(),
            nadi9_text="   ",
            evidence=[make_evidence("E-001")],
            hypotheses=[make_hypothesis()],
            verification=make_verification(),
            confidence=ConfidenceLevel.HIGH,
            confidence_reason="Supported by evidence.",
        )
    except SubtitleDecisionError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError(
            "Expected SubtitleDecisionError"
        )


def test_builder_rejects_empty_confidence_reason():
    builder = SubtitleDecisionBuilder()

    try:
        builder.build(
            episode=make_episode(),
            nadi9_text="Namaskar.",
            evidence=[make_evidence("E-001")],
            hypotheses=[make_hypothesis()],
            verification=make_verification(),
            confidence=ConfidenceLevel.HIGH,
            confidence_reason="   ",
        )
    except SubtitleDecisionError as exc:
        assert "Confidence reason" in str(exc)
    else:
        raise AssertionError(
            "Expected SubtitleDecisionError"
        )