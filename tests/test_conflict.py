from nadi9.domain.models import (
    EvidenceRecord,
    Hypothesis,
    Provenance,
)
from nadi9.graph.conflict import (
    ConflictDetectionError,
    ConflictDetector,
)


def make_evidence(
    evidence_id: str,
    metadata: dict | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-001",
        evidence_type="dictionary_a",
        content=f"Evidence content for {evidence_id}",
        metadata=metadata or {},
        provenance=Provenance(
            original_file="test.json",
            location="entry-1",
            extraction_method="fixture",
        ),
    )


def make_hypothesis(
    supporting_evidence: list[str],
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="H-001",
        category="translation",
        statement="Test hypothesis.",
        supporting_evidence=supporting_evidence,
        counterexamples=[],
        status="proposed",
        confidence="medium",
    )


def test_detector_finds_explicit_conflict():
    detector = ConflictDetector()

    evidence = [
        make_evidence(
            "E-001",
            {
                "conflicts_with": ["E-002"],
            },
        ),
        make_evidence("E-002"),
    ]

    hypothesis = make_hypothesis(
        ["E-001", "E-002"],
    )

    conflicts = detector.detect(
        hypothesis,
        evidence,
    )

    assert len(conflicts) == 1
    assert conflicts[0].conflict_id == (
        "CONFLICT-E-001-E-002"
    )
    assert conflicts[0].evidence_ids == [
        "E-001",
        "E-002",
    ]
    assert conflicts[0].resolved is False
    assert conflicts[0].resolution is None


def test_detector_is_symmetric():
    detector = ConflictDetector()

    evidence = [
        make_evidence(
            "E-001",
            {
                "conflicts_with": ["E-002"],
            },
        ),
        make_evidence("E-002"),
    ]

    hypothesis = make_hypothesis(
        ["E-002", "E-001"],
    )

    conflicts = detector.detect(
        hypothesis,
        evidence,
    )

    assert len(conflicts) == 1
    assert conflicts[0].conflict_id == (
        "CONFLICT-E-001-E-002"
    )


def test_detector_detects_reverse_conflict_marker():
    detector = ConflictDetector()

    evidence = [
        make_evidence("E-001"),
        make_evidence(
            "E-002",
            {
                "conflicts_with": ["E-001"],
            },
        ),
    ]

    hypothesis = make_hypothesis(
        ["E-001", "E-002"],
    )

    conflicts = detector.detect(
        hypothesis,
        evidence,
    )

    assert len(conflicts) == 1


def test_detector_ignores_unrelated_evidence():
    detector = ConflictDetector()

    evidence = [
        make_evidence(
            "E-001",
            {
                "conflicts_with": ["E-002"],
            },
        ),
        make_evidence("E-002"),
        make_evidence("E-003"),
    ]

    hypothesis = make_hypothesis(
        ["E-001", "E-003"],
    )

    conflicts = detector.detect(
        hypothesis,
        evidence,
    )

    assert conflicts == []


def test_detector_does_not_duplicate_conflict():
    detector = ConflictDetector()

    evidence = [
        make_evidence(
            "E-001",
            {
                "conflicts_with": ["E-002"],
            },
        ),
        make_evidence(
            "E-002",
            {
                "conflicts_with": ["E-001"],
            },
        ),
    ]

    hypothesis = make_hypothesis(
        ["E-001", "E-002"],
    )

    conflicts = detector.detect(
        hypothesis,
        evidence,
    )

    assert len(conflicts) == 1


def test_detector_ignores_conflict_with_uncited_evidence():
    detector = ConflictDetector()

    evidence = [
        make_evidence(
            "E-001",
            {
                "conflicts_with": ["E-002"],
            },
        ),
        make_evidence("E-002"),
    ]

    hypothesis = make_hypothesis(
        ["E-001"],
    )

    conflicts = detector.detect(
        hypothesis,
        evidence,
    )

    assert conflicts == []


def test_detector_requires_evidence():
    detector = ConflictDetector()

    hypothesis = make_hypothesis(
        ["E-001"],
    )

    try:
        detector.detect(
            hypothesis,
            [],
        )
    except ConflictDetectionError as exc:
        assert "without evidence" in str(exc)
    else:
        raise AssertionError(
            "Expected ConflictDetectionError"
        )


def test_detector_returns_no_conflicts_for_normal_evidence():
    detector = ConflictDetector()

    evidence = [
        make_evidence("E-001"),
        make_evidence("E-002"),
    ]

    hypothesis = make_hypothesis(
        ["E-001", "E-002"],
    )

    conflicts = detector.detect(
        hypothesis,
        evidence,
    )

    assert conflicts == []