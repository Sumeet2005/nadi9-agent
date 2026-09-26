import pytest

from nadi9.budget import BudgetManager
from nadi9.domain.errors import BudgetExceededError
from nadi9.domain.models import EvidenceRecord, Provenance
from nadi9.ingestion import EvidenceRetriever


def make_evidence(
    evidence_id: str,
    content: str,
    metadata: dict | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="SRC-001",
        evidence_type="dictionary_a",
        content=content,
        metadata=metadata or {},
        provenance=Provenance(
            original_file="test.json",
            location="test",
            extraction_method="fixture",
        ),
    )


def test_retriever_returns_matching_evidence():
    evidence = [
        make_evidence(
            "E-001",
            "formal greeting used by elders",
        ),
        make_evidence(
            "E-002",
            "food vocabulary for cooking",
        ),
    ]

    retriever = EvidenceRetriever(evidence)

    results = retriever.search("formal greeting")

    assert len(results) >= 1
    assert results[0].evidence.evidence_id == "E-001"
    assert results[0].score > 0


def test_retriever_ranks_stronger_match_first():
    evidence = [
        make_evidence(
            "E-001",
            "greeting",
        ),
        make_evidence(
            "E-002",
            "formal greeting used by elders",
        ),
    ]

    retriever = EvidenceRetriever(evidence)

    results = retriever.search("formal greeting elders")

    assert results[0].evidence.evidence_id == "E-002"
    assert results[0].score > results[1].score


def test_retriever_respects_top_k():
    evidence = [
        make_evidence("E-001", "greeting example"),
        make_evidence("E-002", "greeting example two"),
        make_evidence("E-003", "greeting example three"),
    ]

    retriever = EvidenceRetriever(evidence)

    results = retriever.search(
        "greeting",
        top_k=2,
    )

    assert len(results) == 2


def test_retriever_respects_min_score():
    evidence = [
        make_evidence(
            "E-001",
            "formal greeting elders",
        ),
        make_evidence(
            "E-002",
            "food cooking",
        ),
    ]

    retriever = EvidenceRetriever(evidence)

    results = retriever.search(
        "formal greeting elders",
        min_score=0.5,
    )

    assert len(results) == 1
    assert results[0].evidence.evidence_id == "E-001"


def test_empty_query_returns_no_results():
    evidence = [
        make_evidence(
            "E-001",
            "formal greeting",
        ),
    ]

    retriever = EvidenceRetriever(evidence)

    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_retriever_uses_metadata_for_matching():
    evidence = [
        make_evidence(
            "E-001",
            "word meaning",
            metadata={
                "category": "kinship",
                "speaker": "elder",
            },
        ),
    ]

    retriever = EvidenceRetriever(evidence)

    results = retriever.search("kinship")

    assert len(results) == 1
    assert results[0].evidence.evidence_id == "E-001"


def test_retriever_results_are_deterministically_ordered():
    evidence = [
        make_evidence("E-002", "greeting"),
        make_evidence("E-001", "greeting"),
    ]

    retriever = EvidenceRetriever(evidence)

    results = retriever.search("greeting")

    assert [item.evidence.evidence_id for item in results] == [
        "E-001",
        "E-002",
    ]


def test_invalid_top_k_is_rejected():
    retriever = EvidenceRetriever([])

    try:
        retriever.search("greeting", top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_invalid_min_score_is_rejected():
    retriever = EvidenceRetriever([])

    try:
        retriever.search("greeting", min_score=1.5)
    except ValueError as exc:
        assert "min_score" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_retriever_consumes_tool_call_budget_and_raises():
    evidence = [make_evidence("E-001", "formal greeting")]
    budget = BudgetManager(max_model_calls=10, max_tool_calls=1)

    retriever = EvidenceRetriever(evidence, budget=budget)
    results = retriever.search("formal greeting")
    assert len(results) == 1
    assert budget.tool_calls_used == 1

    with pytest.raises(BudgetExceededError) as exc_info:
        retriever.search("formal greeting")
    assert "Tool-call budget exceeded" in str(exc_info.value)