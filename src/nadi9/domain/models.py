from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ConfidenceLevel,
    DecisionStatus,
    EvidenceStatus,
    EvidenceType,
    HypothesisStatus,
    ReviewPriority,
    VerificationStatus,
)


class SubtitlePriority(BaseModel):
    """Prioritization and risk scoring breakdown for one subtitle line."""

    model_config = ConfigDict(extra="forbid")

    subtitle_id: str
    priority: ReviewPriority = ReviewPriority.MEDIUM
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_reasons: list[str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    requires_deep_reasoning: bool = False


class EpisodePlan(BaseModel):
    """Structured execution plan generated prior to processing episode lines."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    priorities: list[SubtitlePriority] = Field(default_factory=list)
    risky_subtitles: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    planned_checks: dict[str, list[str]] = Field(default_factory=dict)
    status: str = "planned"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Provenance(BaseModel):
    """Identifies where an evidence item came from."""

    model_config = ConfigDict(extra="forbid")

    original_file: str
    location: str | None = None
    extraction_method: str
    source_reference: str | None = None


class Source(BaseModel):
    """Metadata describing an evidence source."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: EvidenceType
    name: str
    author: str | None = None
    date: str | None = None
    scope: str | None = None
    declared_reliability: str | None = None
    provenance: Provenance


class EvidenceRecord(BaseModel):
    """A normalized piece of evidence available to the reasoning system."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str
    evidence_type: EvidenceType
    content: str = Field(min_length=1)
    status: EvidenceStatus = EvidenceStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance


class ExampleEvidence(EvidenceRecord):
    """An approved source/Nadi-9 example."""

    evidence_type: EvidenceType = EvidenceType.APPROVED_EXAMPLE

    source_text: str = Field(min_length=1)
    nadi9_text: str = Field(min_length=1)
    speaker: str | None = None
    scene: str | None = None


class DictionaryEntry(EvidenceRecord):
    """A normalized dictionary entry."""

    dictionary: str
    term: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    part_of_speech: str | None = None
    notes: str | None = None


class AudioSegment(EvidenceRecord):
    """A normalized segment from an interview."""

    evidence_type: EvidenceType = EvidenceType.AUDIO_INTERVIEW

    interview_id: str
    speaker: str | None = None
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    transcript: str = Field(min_length=1)


class EpisodeLine(BaseModel):
    """A source-language episode line that can become a subtitle decision."""

    model_config = ConfigDict(extra="forbid")

    subtitle_id: str
    source_text: str = Field(min_length=1)
    speaker: str | None = None
    scene_id: str | None = None
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    context: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """A candidate linguistic or translation rule derived from evidence."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    category: str
    statement: str = Field(min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VerificationResult(BaseModel):
    """Result produced by independent verification."""

    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    checks: dict[str, bool] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    evidence_checked: list[str] = Field(default_factory=list)
    verifier_notes: str | None = None


class Conflict(BaseModel):
    """Represents disagreement between evidence or interpretations."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=2)
    resolution: str | None = None
    resolved: bool = False


class ReviewItem(BaseModel):
    """A focused human-review request."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    subtitle_id: str
    reason: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    priority: ReviewPriority = ReviewPriority.MEDIUM
    resolved: bool = False


class HumanReviewAction(BaseModel):
    """Auditable record of a human-in-the-loop decision action."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    review_id: str
    subtitle_id: str
    action_type: str  # "approve", "reject", "correct"
    actor: str = "human_reviewer"
    reason: str | None = None
    corrected_text: str | None = None
    original_nadi9_text: str
    final_status: DecisionStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubtitleDecision(BaseModel):
    """Complete, auditable decision for one subtitle."""

    model_config = ConfigDict(extra="forbid")

    subtitle_id: str
    source_text: str = Field(min_length=1)
    nadi9_text: str = Field(min_length=1)
    status: DecisionStatus = DecisionStatus.DRAFT

    evidence_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)

    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    confidence_reason: str = Field(min_length=1)

    assumptions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    verification: VerificationResult | None = None

    human_review_required: bool = False
    review_question: str | None = None
    abstained: bool = False

    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LearnedRule(BaseModel):
    """A linguistic or translation rule learned or confirmed from evidence or human review."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    category: str
    statement: str = Field(min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list)
    affected_subtitle_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    supersedes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))