from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nadi9.domain.enums import (
    ConfidenceLevel,
    DecisionStatus,
    VerificationStatus,
)
from nadi9.domain.models import (
    Conflict,
    EvidenceRecord,
    Hypothesis,
    ReviewItem,
    SubtitleDecision,
    VerificationResult,
)
from nadi9.graph.state import AgentState


class EvidenceAuditDetail(BaseModel):
    """Audit summary of an evidence item used in reasoning."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str
    evidence_type: str
    content: str
    status: str
    supported_hypotheses: list[str] = Field(default_factory=list)


class HypothesisAuditDetail(BaseModel):
    """Audit summary of a generated hypothesis."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    category: str
    statement: str
    supporting_evidence: list[str]
    counterexamples: list[str]
    status: str
    confidence: ConfidenceLevel


class DecisionAuditDetail(BaseModel):
    """Complete structured audit detail for one subtitle decision."""

    model_config = ConfigDict(extra="forbid")

    subtitle_id: str
    source_text: str
    nadi9_text: str
    status: DecisionStatus
    confidence: ConfidenceLevel
    confidence_reason: str
    human_review_required: bool
    review_question: str | None = None

    verification: VerificationResult | None = None
    evidence: list[EvidenceAuditDetail] = Field(default_factory=list)
    hypotheses: list[HypothesisAuditDetail] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)


class AuditReport(BaseModel):
    """Overall auditable report for an episode run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    episode_id: str
    total_subtitles_processed: int
    accepted_count: int
    human_review_count: int
    decisions: list[DecisionAuditDetail] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)
    budget_summary: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit report to a plain Python dictionary."""
        return self.model_dump(mode="json")


def build_decision_audit(
    *,
    decision: SubtitleDecision,
    evidence: list[EvidenceRecord],
    hypotheses: list[Hypothesis],
    conflicts: list[Conflict],
    review_items: list[ReviewItem],
) -> DecisionAuditDetail:
    """Build a detailed audit record for a single subtitle decision."""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    cited_evidence_ids = decision.evidence_ids

    evidence_details: list[EvidenceAuditDetail] = []
    for ev_id in cited_evidence_ids:
        if ev_id in evidence_by_id:
            record = evidence_by_id[ev_id]
            supporting_h = [
                h.hypothesis_id
                for h in hypotheses
                if ev_id in h.supporting_evidence
            ]
            evidence_details.append(
                EvidenceAuditDetail(
                    evidence_id=record.evidence_id,
                    source_id=record.source_id,
                    evidence_type=record.evidence_type.value,
                    content=record.content,
                    status=record.status.value,
                    supported_hypotheses=supporting_h,
                )
            )

    hypothesis_details = [
        HypothesisAuditDetail(
            hypothesis_id=h.hypothesis_id,
            category=h.category,
            statement=h.statement,
            supporting_evidence=h.supporting_evidence,
            counterexamples=h.counterexamples,
            status=h.status.value,
            confidence=h.confidence,
        )
        for h in hypotheses
        if h.hypothesis_id in decision.hypothesis_ids
    ]

    conflict_details = [
        c for c in conflicts if c.conflict_id in decision.conflicts
    ]

    subtitle_reviews = [
        r for r in review_items if r.subtitle_id == decision.subtitle_id
    ]

    return DecisionAuditDetail(
        subtitle_id=decision.subtitle_id,
        source_text=decision.source_text,
        nadi9_text=decision.nadi9_text,
        status=decision.status,
        confidence=decision.confidence,
        confidence_reason=decision.confidence_reason,
        human_review_required=decision.human_review_required,
        review_question=decision.review_question,
        verification=decision.verification,
        evidence=evidence_details,
        hypotheses=hypothesis_details,
        conflicts=conflict_details,
        review_items=subtitle_reviews,
    )


def build_episode_audit_report(state: AgentState) -> AuditReport:
    """Build an overall audit report from a completed AgentState."""

    run_id = state.get("run_id", "unknown_run")
    episode_id = state.get("episode_id", "unknown_episode")

    decisions = state.get("subtitle_decisions", [])
    evidence = state.get("evidence", [])
    hypotheses = state.get("hypotheses", [])
    conflicts = state.get("conflicts", [])
    review_items = state.get("review_items", [])
    budget = state.get("budget", {})

    decision_audits = [
        build_decision_audit(
            decision=d,
            evidence=evidence,
            hypotheses=hypotheses,
            conflicts=conflicts,
            review_items=review_items,
        )
        for d in decisions
    ]

    accepted_count = sum(
        1 for d in decisions if d.status == DecisionStatus.ACCEPTED
    )
    human_review_count = sum(
        1 for d in decisions if d.human_review_required or d.status == DecisionStatus.REVIEW_REQUIRED
    )

    budget_summary = {
        "model_calls_used": budget.get("model_calls_used", 0),
        "tool_calls_used": budget.get("tool_calls_used", 0),
        "max_model_calls": budget.get("max_model_calls", 0),
        "max_tool_calls": budget.get("max_tool_calls", 0),
    }

    return AuditReport(
        run_id=run_id,
        episode_id=episode_id,
        total_subtitles_processed=len(decisions),
        accepted_count=accepted_count,
        human_review_count=human_review_count,
        decisions=decision_audits,
        review_items=review_items,
        budget_summary=budget_summary,
        errors=list(state.get("errors", [])),
    )
