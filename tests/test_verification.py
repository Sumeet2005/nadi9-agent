from nadi9.domain.models import (
    EvidenceRecord,
    Hypothesis,
    Provenance,
)
from nadi9.graph.verification import (
    HypothesisVerifier,
    VerificationError,
)


def make_evidence(
    evidence_id: str,
    content: str = "formal greeting",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-001",
        evidence_type="dictionary_a",
        content=content,
        provenance=Provenance(
            original_file="test.json",
            location="entry-1",
            extraction_method="fixture",
        ),
    )


def make_hypothesis(
    supporting_evidence: list[str],
    statement: str = "Formal greeting.",
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="H-001",
        category="greeting",
        statement=statement,
        supporting_evidence=supporting_evidence,
        counterexamples=[],
        status="proposed",
        confidence="medium",
    )


def test_verifier_passes_supported_hypothesis():
    verifier = HypothesisVerifier()

    hypothesis = make_hypothesis(["E-001"])
    evidence = [
        make_evidence("E-001"),
    ]

    result = verifier.verify(
        hypothesis,
        evidence,
    )

    assert result.status == "passed"
    assert result.checks["supporting_evidence_exists"] is True
    assert result.checks["has_supporting_evidence"] is True
    assert result.checks["statement_present"] is True
    assert result.evidence_checked == ["E-001"]
    assert result.failures == []


def test_verifier_rejects_missing_evidence():
    verifier = HypothesisVerifier()

    hypothesis = make_hypothesis(["E-999"])
    evidence = [
        make_evidence("E-001"),
    ]

    result = verifier.verify(
        hypothesis,
        evidence,
    )

    assert result.status == "failed"
    assert result.checks["supporting_evidence_exists"] is False
    assert "E-999" in result.failures[0]
    assert result.evidence_checked == []


def test_verifier_rejects_hypothesis_without_supporting_evidence():
    verifier = HypothesisVerifier()

    hypothesis = make_hypothesis([])
    evidence = [
        make_evidence("E-001"),
    ]

    result = verifier.verify(
        hypothesis,
        evidence,
    )

    assert result.status == "failed"
    assert result.checks["has_supporting_evidence"] is False
    assert (
        "does not cite supporting evidence"
        in result.failures[0]
    )


def test_verifier_rejects_empty_statement():
    verifier = HypothesisVerifier()

    hypothesis = make_hypothesis(
        ["E-001"],
        statement="   ",
    )

    evidence = [
        make_evidence("E-001"),
    ]

    result = verifier.verify(
        hypothesis,
        evidence,
    )

    assert result.status == "failed"
    assert result.checks["statement_present"] is False
    assert "statement is empty" in result.failures[0]


def test_verifier_checks_only_existing_evidence():
    verifier = HypothesisVerifier()

    hypothesis = make_hypothesis(
        ["E-001", "E-002"],
    )

    evidence = [
        make_evidence("E-001"),
    ]

    result = verifier.verify(
        hypothesis,
        evidence,
    )

    assert result.status == "failed"
    assert result.evidence_checked == ["E-001"]


def test_verifier_requires_evidence():
    verifier = HypothesisVerifier()

    hypothesis = make_hypothesis(["E-001"])

    try:
        verifier.verify(
            hypothesis,
            [],
        )
    except VerificationError as exc:
        assert "without evidence" in str(exc)
    else:
        raise AssertionError(
            "Expected VerificationError"
        )