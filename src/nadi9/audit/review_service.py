import json
from pathlib import Path
from typing import Any
import uuid

from pydantic import ValidationError

from nadi9.domain.enums import DecisionStatus
from nadi9.domain.models import HumanReviewAction, ReviewItem, SubtitleDecision


class ReviewError(Exception):
    """Base exception for human-in-the-loop review operations."""


class ReviewNotFoundError(ReviewError):
    """Raised when a specified review ID is not found."""


class ReviewAlreadyResolvedError(ReviewError):
    """Raised when an action is attempted on an already-resolved review item."""


class RunArtifactError(ReviewError):
    """Raised when run directory or required artifacts are missing or malformed."""


class ReviewManager:
    """Service handling human-in-the-loop review actions over output artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists() or not self.run_dir.is_dir():
            raise RunArtifactError(
                f"Run directory '{self.run_dir}' does not exist or is not a directory."
            )

        self.review_items_file = self.run_dir / "review_items.json"
        self.decisions_file = self.run_dir / "decisions.json"
        self.report_file = self.run_dir / "report.json"
        self.actions_file = self.run_dir / "review_actions.json"

        if not self.review_items_file.exists():
            raise RunArtifactError(
                f"review_items.json is missing from run directory '{self.run_dir}'."
            )
        if not self.decisions_file.exists():
            raise RunArtifactError(
                f"decisions.json is missing from run directory '{self.run_dir}'."
            )

    def list_reviews(self, include_resolved: bool = False) -> list[dict[str, Any]]:
        """List review items with decision context."""

        review_items, decisions_by_sub = self._load_artifacts()

        results: list[dict[str, Any]] = []
        for item in review_items:
            if not include_resolved and item.resolved:
                continue

            decision = decisions_by_sub.get(item.subtitle_id)
            results.append(
                {
                    "review_id": item.review_id,
                    "subtitle_id": item.subtitle_id,
                    "reason": item.reason,
                    "question": item.question,
                    "priority": item.priority.value,
                    "confidence": item.confidence.value,
                    "resolved": item.resolved,
                    "current_nadi9_text": decision.nadi9_text if decision else None,
                    "decision_status": (
                        decision.status.value if decision else "unknown"
                    ),
                    "evidence_ids": item.evidence_ids,
                    "conflict_ids": item.conflict_ids,
                }
            )

        return results

    def approve(
        self,
        review_id: str,
        actor: str = "human_reviewer",
    ) -> HumanReviewAction:
        """Approve an AI decision without altering original subtitle text."""

        review_items, decisions_by_sub = self._load_artifacts()

        target_item = self._find_item(review_items, review_id)
        decision = self._find_decision(decisions_by_sub, target_item.subtitle_id)

        target_item.resolved = True
        decision.status = DecisionStatus.ACCEPTED
        decision.human_review_required = False
        decision.confidence_reason = (
            f"Approved by human reviewer ({actor}). "
            f"Original reason: {decision.confidence_reason}"
        )

        action = HumanReviewAction(
            action_id=f"ACT-{uuid.uuid4().hex[:8]}",
            review_id=review_id,
            subtitle_id=decision.subtitle_id,
            action_type="approve",
            actor=actor,
            reason="Approved AI candidate subtitle.",
            original_nadi9_text=decision.nadi9_text,
            final_status=DecisionStatus.ACCEPTED,
        )

        self._save_artifacts(review_items, decisions_by_sub, action)
        return action

    def reject(
        self,
        review_id: str,
        reason: str,
        actor: str = "human_reviewer",
    ) -> HumanReviewAction:
        """Reject an AI decision recording a mandatory rejection reason."""

        if not reason.strip():
            raise ValueError("Rejection reason cannot be empty.")

        review_items, decisions_by_sub = self._load_artifacts()

        target_item = self._find_item(review_items, review_id)
        decision = self._find_decision(decisions_by_sub, target_item.subtitle_id)

        target_item.resolved = True
        decision.status = DecisionStatus.REJECTED
        decision.human_review_required = False
        decision.confidence_reason = f"Rejected by human reviewer ({actor}): {reason.strip()}"

        action = HumanReviewAction(
            action_id=f"ACT-{uuid.uuid4().hex[:8]}",
            review_id=review_id,
            subtitle_id=decision.subtitle_id,
            action_type="reject",
            actor=actor,
            reason=reason.strip(),
            original_nadi9_text=decision.nadi9_text,
            final_status=DecisionStatus.REJECTED,
        )

        self._save_artifacts(review_items, decisions_by_sub, action)
        return action

    def correct(
        self,
        review_id: str,
        text: str,
        reason: str | None = None,
        actor: str = "human_reviewer",
    ) -> HumanReviewAction:
        """Supply human-corrected subtitle text preserving original AI text in action log."""

        if not text.strip():
            raise ValueError("Corrected text cannot be empty.")

        review_items, decisions_by_sub = self._load_artifacts()

        target_item = self._find_item(review_items, review_id)
        decision = self._find_decision(decisions_by_sub, target_item.subtitle_id)

        original_text = decision.nadi9_text

        target_item.resolved = True
        decision.nadi9_text = text.strip()
        decision.status = DecisionStatus.ACCEPTED
        decision.human_review_required = False
        decision.confidence_reason = (
            f"Corrected by human reviewer ({actor}). "
            + (f"Note: {reason.strip()}" if reason and reason.strip() else "")
        )

        action = HumanReviewAction(
            action_id=f"ACT-{uuid.uuid4().hex[:8]}",
            review_id=review_id,
            subtitle_id=decision.subtitle_id,
            action_type="correct",
            actor=actor,
            reason=reason.strip() if reason else "Human correction supplied.",
            corrected_text=text.strip(),
            original_nadi9_text=original_text,
            final_status=DecisionStatus.ACCEPTED,
        )

        self._save_artifacts(review_items, decisions_by_sub, action)
        return action

    def _load_artifacts(
        self,
    ) -> tuple[list[ReviewItem], dict[str, SubtitleDecision]]:
        """Safely load review_items.json and decisions.json."""

        try:
            raw_reviews = json.loads(
                self.review_items_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise RunArtifactError(
                f"Malformed JSON in review_items.json: {exc}"
            ) from exc

        try:
            raw_decisions = json.loads(
                self.decisions_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise RunArtifactError(
                f"Malformed JSON in decisions.json: {exc}"
            ) from exc

        try:
            review_items = [
                ReviewItem.model_validate(r) for r in raw_reviews
            ]
        except ValidationError as exc:
            raise RunArtifactError(
                f"Invalid ReviewItem schema in review_items.json: {exc}"
            ) from exc

        valid_decision_keys = set(SubtitleDecision.model_fields.keys())
        parsed_decisions: list[SubtitleDecision] = []

        for d in raw_decisions:
            if isinstance(d, dict):
                d_copy = dict(d)
                if "conflicts" in d_copy and isinstance(d_copy["conflicts"], list):
                    conf_ids = []
                    for c in d_copy["conflicts"]:
                        if isinstance(c, dict) and "conflict_id" in c:
                            conf_ids.append(c["conflict_id"])
                        elif isinstance(c, str):
                            conf_ids.append(c)
                    d_copy["conflicts"] = conf_ids

                filtered = {k: v for k, v in d_copy.items() if k in valid_decision_keys}
                try:
                    parsed_decisions.append(SubtitleDecision.model_validate(filtered))
                except ValidationError as exc:
                    raise RunArtifactError(
                        f"Invalid SubtitleDecision schema in decisions.json: {exc}"
                    ) from exc
            else:
                try:
                    parsed_decisions.append(SubtitleDecision.model_validate(d))
                except ValidationError as exc:
                    raise RunArtifactError(
                        f"Invalid SubtitleDecision schema in decisions.json: {exc}"
                    ) from exc

        decisions_by_sub = {d.subtitle_id: d for d in parsed_decisions}
        return review_items, decisions_by_sub

    @staticmethod
    def _find_item(items: list[ReviewItem], review_id: str) -> ReviewItem:
        for item in items:
            if item.review_id == review_id:
                if item.resolved:
                    raise ReviewAlreadyResolvedError(
                        f"Review '{review_id}' has already been resolved."
                    )
                return item
        raise ReviewNotFoundError(f"Review ID '{review_id}' was not found.")

    @staticmethod
    def _find_decision(
        decisions_by_sub: dict[str, SubtitleDecision], subtitle_id: str
    ) -> SubtitleDecision:
        if subtitle_id in decisions_by_sub:
            return decisions_by_sub[subtitle_id]
        raise RunArtifactError(
            f"Corresponding decision for subtitle '{subtitle_id}' not found."
        )

    def _save_artifacts(
        self,
        review_items: list[ReviewItem],
        decisions_by_sub: dict[str, SubtitleDecision],
        action: HumanReviewAction,
    ) -> None:
        """Atomic write update of review items, decisions, actions, and report."""

        actions: list[dict[str, Any]] = []
        if self.actions_file.exists():
            try:
                actions = json.loads(self.actions_file.read_text(encoding="utf-8"))
            except Exception:
                actions = []

        actions.append(action.model_dump(mode="json"))

        items_data = [r.model_dump(mode="json") for r in review_items]
        self.review_items_file.write_text(
            json.dumps(items_data, indent=2), encoding="utf-8"
        )

        decisions_data = [d.model_dump(mode="json") for d in decisions_by_sub.values()]
        self.decisions_file.write_text(
            json.dumps(decisions_data, indent=2), encoding="utf-8"
        )

        self.actions_file.write_text(
            json.dumps(actions, indent=2), encoding="utf-8"
        )

        if self.report_file.exists():
            try:
                report_raw = json.loads(self.report_file.read_text(encoding="utf-8"))
                report_raw["review_items"] = items_data
                accepted_count = sum(
                    1 for d in decisions_by_sub.values() if d.status == DecisionStatus.ACCEPTED
                )
                review_count = sum(
                    1 for d in review_items if not d.resolved
                )
                report_raw["accepted_count"] = accepted_count
                report_raw["human_review_count"] = review_count

                for dec_audit in report_raw.get("decisions", []):
                    sub_id = dec_audit.get("subtitle_id")
                    if sub_id in decisions_by_sub:
                        updated_dec = decisions_by_sub[sub_id]
                        dec_audit["nadi9_text"] = updated_dec.nadi9_text
                        dec_audit["status"] = (
                            updated_dec.status.value
                            if hasattr(updated_dec.status, "value")
                            else str(updated_dec.status)
                        )
                        dec_audit["human_review_required"] = updated_dec.human_review_required
                        dec_audit["confidence_reason"] = updated_dec.confidence_reason

                self.report_file.write_text(
                    json.dumps(report_raw, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
