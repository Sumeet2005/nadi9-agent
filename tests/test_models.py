import pytest
from pydantic import ValidationError

from nadi9.domain.enums import (
    ConfidenceLevel,
    DecisionStatus,
    EvidenceType,
)
from nadi9.domain.models import (
    Provenance,
    SubtitleDecision,
)


def test_provenance_requires_original_file():
    provenance = Provenance(
        original_file="grammar.pdf",
        extraction_method="pdf_text",
    )

    assert provenance.original_file == "grammar.pdf"


def test_subtitle_decision_accepts_valid_data():
    decision = SubtitleDecision(
        subtitle_id="S001",
        source_text="Hello",
        nadi9_text="Demo translation",
        status=DecisionStatus.TRANSLATED,
        evidence_ids=["EX-001"],
        confidence=ConfidenceLevel.MEDIUM,
        confidence_reason="Supported by one approved example.",
    )

    assert decision.subtitle_id == "S001"
    assert decision.evidence_ids == ["EX-001"]


def test_subtitle_decision_rejects_empty_source_text():
    with pytest.raises(ValidationError):
        SubtitleDecision(
            subtitle_id="S001",
            source_text="",
            nadi9_text="Demo translation",
            confidence_reason="Test reason",
        )


def test_evidence_type_is_controlled():
    assert EvidenceType.GRAMMAR.value == "grammar"