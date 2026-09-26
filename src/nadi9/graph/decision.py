from nadi9.domain.enums import (
    ConfidenceLevel,
    DecisionStatus,
    VerificationStatus,
)
from nadi9.domain.models import (
    Conflict,
    EpisodeLine,
    EvidenceRecord,
    Hypothesis,
    SubtitleDecision,
    VerificationResult,
)


class SubtitleDecisionError(ValueError):
    """Raised when a subtitle decision cannot be created."""


class SubtitleDecisionBuilder:
    """Builds an auditable subtitle decision from verified inputs."""

    def build(
        self,
        *,
        episode: EpisodeLine,
        nadi9_text: str,
        evidence: list[EvidenceRecord],
        hypotheses: list[Hypothesis],
        verification: VerificationResult,
        conflicts: list[Conflict] | None = None,
        confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN,
        confidence_reason: str = "",
        assumptions: list[str] | None = None,
    ) -> SubtitleDecision:
        """Build a validated subtitle decision."""

        conflicts = conflicts or []
        assumptions = assumptions or []

        if not nadi9_text.strip():
            raise SubtitleDecisionError(
                "Nadi-9 subtitle text cannot be empty."
            )

        if not confidence_reason.strip():
            raise SubtitleDecisionError(
                "Confidence reason cannot be empty."
            )

        evidence_by_id = {
            item.evidence_id: item
            for item in evidence
        }

        evidence_ids = self._collect_evidence_ids(
            hypotheses=hypotheses,
            evidence_by_id=evidence_by_id,
        )

        hypothesis_ids = [
            hypothesis.hypothesis_id
            for hypothesis in hypotheses
        ]

        conflict_ids = [
            conflict.conflict_id
            for conflict in conflicts
        ]

        human_review_required = self._requires_human_review(
            verification=verification,
            conflicts=conflicts,
        )

        review_question = self._build_review_question(
            verification=verification,
            conflicts=conflicts,
            evidence=evidence,
        )

        status = self._determine_status(
            verification=verification,
            conflicts=conflicts,
            human_review_required=human_review_required,
        )

        abstained = nadi9_text == "Translation unavailable: insufficient supporting evidence."

        return SubtitleDecision(
            subtitle_id=episode.subtitle_id,
            source_text=episode.source_text,
            nadi9_text=nadi9_text,
            status=status,
            evidence_ids=evidence_ids,
            hypothesis_ids=hypothesis_ids,
            confidence=confidence,
            confidence_reason=confidence_reason,
            assumptions=assumptions,
            conflicts=conflict_ids,
            verification=verification,
            human_review_required=human_review_required,
            review_question=review_question,
            abstained=abstained,
        )

    @staticmethod
    def _collect_evidence_ids(
        *,
        hypotheses: list[Hypothesis],
        evidence_by_id: dict[str, EvidenceRecord],
    ) -> list[str]:
        """Collect unique evidence IDs cited by hypotheses."""

        evidence_ids: list[str] = []

        for hypothesis in hypotheses:
            for evidence_id in hypothesis.supporting_evidence:
                if evidence_id not in evidence_by_id:
                    continue

                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)

        return evidence_ids

    @staticmethod
    def _requires_human_review(
        *,
        verification: VerificationResult,
        conflicts: list[Conflict],
    ) -> bool:
        """Determine whether a human must review the decision."""

        if verification.status != VerificationStatus.PASSED:
            return True

        if any(not conflict.resolved for conflict in conflicts):
            return True

        return False

    @staticmethod
    def _build_review_question(
        *,
        verification: VerificationResult,
        conflicts: list[Conflict],
        evidence: list[EvidenceRecord] | None = None,
    ) -> str | None:
        """Create a focused, precise review question when required."""

        unresolved_conflicts = [
            conflict for conflict in (conflicts or []) if not conflict.resolved
        ]

        if unresolved_conflicts:
            conflict_ids = ", ".join(conflict.conflict_id for conflict in unresolved_conflicts)
            return (
                "Please resolve the conflicting evidence and confirm which interpretation "
                f"should be used for this subtitle. Conflict IDs: {conflict_ids}."
            )

        if verification.status != VerificationStatus.PASSED:
            failures_str = "; ".join(verification.failures) if verification.failures else ""
            if "has_supporting_evidence" in verification.checks and not verification.checks["has_supporting_evidence"]:
                return "Please provide or confirm an approved Nadi-9 translation because no supporting evidence was found for this subtitle."

            if "proper_name" in failures_str.lower() or "name" in failures_str.lower():
                return "Please verify the proper name in the proposed translation because the independent name-preservation check failed."

            if "number" in failures_str.lower() or "quantity" in failures_str.lower():
                return "Please verify the numeric value because the source and proposed translation contain different quantities."

            if failures_str:
                return f"Please review the proposed translation because independent verification failed: {failures_str}."

            return "Please review this subtitle decision because independent verification did not pass."

        if not evidence:
            return "Please provide or confirm an approved Nadi-9 translation because no supporting evidence was found for this subtitle."

        return None

    @staticmethod
    def _determine_status(
        *,
        verification: VerificationResult,
        conflicts: list[Conflict],
        human_review_required: bool,
    ) -> DecisionStatus:
        """Determine the auditable decision status."""

        if human_review_required:
            return DecisionStatus.REVIEW_REQUIRED

        if verification.status == VerificationStatus.PASSED:
            return DecisionStatus.ACCEPTED

        if conflicts:
            return DecisionStatus.REVIEW_REQUIRED

        return DecisionStatus.DRAFT