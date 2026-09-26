from nadi9.domain.models import EvidenceRecord, Provenance
from nadi9.ingestion import (
    EvidenceRanker,
    EvidenceRankingPolicy,
    RetrievedEvidence,
)


def make_evidence(
    evidence_id: str,
    evidence_type: str = "dictionary_a",
    metadata: dict | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-001",
        evidence_type=evidence_type,
        content="formal greeting used by elders",
        metadata=metadata or {},
        provenance=Provenance(
            original_file="evidence.json",
            location="entry-1",
            extraction_method="fixture",
        ),
    )


def make_result(
    evidence_id: str,
    score: float,
    evidence_type: str = "dictionary_a",
    metadata: dict | None = None,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence=make_evidence(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            metadata=metadata,
        ),
        score=score,
    )


def test_default_policy_source_precedence():
    policy = EvidenceRankingPolicy()
    item = make_result("E-001", 0.5, evidence_type="approved_example")
    # Base 0.5 + approved_example precedence 0.50 + provenance 0.05 = 1.05
    assert policy.score(item) == 1.05


def test_policy_can_add_evidence_type_weight():
    policy = EvidenceRankingPolicy(
        type_weights={
            "expert_note": 0.2,
        },
        provenance_bonus=0.0,
    )

    item = make_result(
        "E-001",
        0.5,
        evidence_type="expert_note",
    )

    assert policy.score(item) == 0.7


def test_policy_can_add_provenance_bonus():
    policy = EvidenceRankingPolicy(
        type_weights={},
        provenance_bonus=0.1,
    )

    item = make_result("E-001", 0.5)

    assert policy.score(item) == 0.6


def test_policy_can_add_reliability_bonus():
    policy = EvidenceRankingPolicy(
        type_weights={},
        provenance_bonus=0.0,
        reliability_bonus={
            "high": 0.2,
        },
    )

    item = make_result(
        "E-001",
        0.5,
        metadata={
            "reliability": "high",
        },
    )

    assert policy.score(item) == 0.7


def test_source_precedence_ranks_higher_authority_first():
    policy = EvidenceRankingPolicy()
    ranker = EvidenceRanker(policy)

    # Dictionary B (newer/returned with search score 0.7) vs Approved Example (lower search score 0.5)
    dict_b_item = make_result("E-DICT", 0.7, evidence_type="dictionary_b")
    approved_item = make_result("E-APP", 0.5, evidence_type="approved_example")

    ranked = ranker.rank([dict_b_item, approved_item])

    # Approved example must rank higher due to source precedence despite lower search score
    assert ranked[0].evidence.evidence_id == "E-APP"


def test_ranker_explain_ranking_provides_auditable_breakdown():
    ranker = EvidenceRanker()
    dict_b_item = make_result("E-DICT", 0.7, evidence_type="dictionary_b")
    approved_item = make_result("E-APP", 0.5, evidence_type="approved_example")

    explanations = ranker.explain_ranking([dict_b_item, approved_item])

    assert len(explanations) == 2
    assert explanations[0]["evidence_id"] == "E-APP"
    assert explanations[0]["rank"] == 1
    assert "type_precedence_weight" in explanations[0]
    assert "precedence_rationale" in explanations[0]


def test_ranker_is_deterministic_for_equal_scores():
    ranker = EvidenceRanker()

    results = [
        make_result("E-002", 0.5),
        make_result("E-001", 0.5),
    ]

    ranked = ranker.rank(results)

    assert [
        item.evidence.evidence_id
        for item in ranked
    ] == [
        "E-001",
        "E-002",
    ]


def test_ranker_respects_top_k():
    ranker = EvidenceRanker()

    results = [
        make_result("E-001", 0.9),
        make_result("E-002", 0.8),
        make_result("E-003", 0.7),
    ]

    ranked = ranker.rank(
        results,
        top_k=2,
    )

    assert len(ranked) == 2
    assert [
        item.evidence.evidence_id
        for item in ranked
    ] == [
        "E-001",
        "E-002",
    ]


def test_ranker_rejects_invalid_top_k():
    ranker = EvidenceRanker()

    try:
        ranker.rank([], top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_unknown_reliability_has_no_bonus():
    policy = EvidenceRankingPolicy(
        type_weights={},
        provenance_bonus=0.0,
        reliability_bonus={
            "high": 0.2,
        },
    )

    item = make_result(
        "E-001",
        0.5,
        metadata={
            "reliability": "unknown",
        },
    )

    assert policy.score(item) == 0.5