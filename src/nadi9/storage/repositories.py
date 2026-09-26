from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from nadi9.domain.enums import DecisionStatus, ReviewPriority
from nadi9.domain.models import (
    HumanReviewAction,
    ReviewItem,
    SubtitleDecision,
)
from nadi9.storage.models import (
    AuditEventModel,
    HumanReviewActionModel,
    ReviewItemModel,
    RunModel,
    SubtitleDecisionModel,
)


class RunRepository:
    """Repository for managing durable Run records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        run_id: str,
        episode_id: str,
        config_snapshot: dict[str, Any] | None = None,
    ) -> RunModel:
        existing = self.get_run(run_id)
        if existing:
            return existing

        run = RunModel(
            run_id=run_id,
            episode_id=episode_id,
            status="CREATED",
            config_snapshot=config_snapshot or {},
            created_at=datetime.now(UTC),
        )
        self.session.add(run)
        self.session.commit()
        return run

    def update_run_status(
        self,
        run_id: str,
        status: str,
        budget_info: dict[str, Any] | None = None,
        processing_counts: dict[str, Any] | None = None,
        error_info: dict[str, Any] | None = None,
    ) -> RunModel | None:
        stmt = select(RunModel).where(RunModel.run_id == run_id)
        run = self.session.scalar(stmt)
        if not run:
            return None

        run.status = status
        if status == "RUNNING" and not run.started_at:
            run.started_at = datetime.now(UTC)
        elif status in ("COMPLETED", "FAILED"):
            run.completed_at = datetime.now(UTC)

        if budget_info:
            run.budget_info = budget_info
        if processing_counts:
            run.processing_counts = processing_counts
        if error_info:
            run.error_info = error_info

        self.session.commit()
        return run

    def get_run(self, run_id: str) -> RunModel | None:
        return self.session.scalar(select(RunModel).where(RunModel.run_id == run_id))

    def list_runs(self) -> list[RunModel]:
        return list(self.session.scalars(select(RunModel).order_by(RunModel.created_at.desc())).all())


class DecisionRepository:
    """Repository for persisting SubtitleDecision entities."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def save_decision(
        self, run_id: str, decision: SubtitleDecision
    ) -> SubtitleDecisionModel:
        stmt = select(SubtitleDecisionModel).where(
            SubtitleDecisionModel.run_id == run_id,
            SubtitleDecisionModel.subtitle_id == decision.subtitle_id,
        )
        existing = self.session.scalar(stmt)

        ver_dict = (
            decision.verification.model_dump(mode="json")
            if decision.verification
            else None
        )

        status_str = (
            decision.status.value
            if hasattr(decision.status, "value")
            else str(decision.status)
        )
        conf_str = (
            decision.confidence.value
            if hasattr(decision.confidence, "value")
            else str(decision.confidence)
        )

        if existing:
            existing.nadi9_text = decision.nadi9_text
            existing.status = status_str
            existing.confidence = conf_str
            existing.confidence_reason = decision.confidence_reason
            existing.evidence_ids_json = decision.evidence_ids
            existing.hypothesis_ids_json = decision.hypothesis_ids
            existing.conflicts_json = decision.conflicts
            existing.human_review_required = decision.human_review_required
            existing.review_question = decision.review_question
            existing.verification_json = ver_dict
            existing.version += 1
            existing.updated_at = datetime.now(UTC)
            model = existing
        else:
            model = SubtitleDecisionModel(
                run_id=run_id,
                subtitle_id=decision.subtitle_id,
                source_text=decision.source_text,
                nadi9_text=decision.nadi9_text,
                status=status_str,
                confidence=conf_str,
                confidence_reason=decision.confidence_reason,
                evidence_ids_json=decision.evidence_ids,
                hypothesis_ids_json=decision.hypothesis_ids,
                conflicts_json=decision.conflicts,
                human_review_required=decision.human_review_required,
                review_question=decision.review_question,
                verification_json=ver_dict,
                version=decision.version,
                created_at=decision.created_at,
                updated_at=datetime.now(UTC),
            )
            self.session.add(model)

        self.session.commit()
        return model

    def list_decisions_for_run(self, run_id: str) -> list[SubtitleDecisionModel]:
        return list(
            self.session.scalars(
                select(SubtitleDecisionModel).where(
                    SubtitleDecisionModel.run_id == run_id
                )
            ).all()
        )


class ReviewRepository:
    """Repository for ReviewItem and HumanReviewAction persistence."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def save_review_item(self, run_id: str, item: ReviewItem) -> ReviewItemModel:
        stmt = select(ReviewItemModel).where(
            ReviewItemModel.review_id == item.review_id
        )
        existing = self.session.scalar(stmt)

        prio_str = (
            item.priority.value
            if hasattr(item.priority, "value")
            else str(item.priority)
        )
        conf_str = (
            item.confidence.value
            if hasattr(item.confidence, "value")
            else str(item.confidence)
        )

        if existing:
            existing.resolved = item.resolved
            existing.priority = prio_str
            existing.confidence = conf_str
            model = existing
        else:
            model = ReviewItemModel(
                review_id=item.review_id,
                run_id=run_id,
                subtitle_id=item.subtitle_id,
                reason=item.reason,
                question=item.question,
                evidence_ids_json=item.evidence_ids,
                conflict_ids_json=item.conflict_ids,
                confidence=conf_str,
                priority=prio_str,
                resolved=item.resolved,
            )
            self.session.add(model)

        self.session.commit()
        return model

    def save_review_action(
        self, run_id: str, action: HumanReviewAction
    ) -> HumanReviewActionModel:
        final_st = (
            action.final_status.value
            if hasattr(action.final_status, "value")
            else str(action.final_status)
        )
        model = HumanReviewActionModel(
            action_id=action.action_id,
            run_id=run_id,
            review_id=action.review_id,
            subtitle_id=action.subtitle_id,
            action_type=action.action_type,
            actor=action.actor,
            reason=action.reason,
            corrected_text=action.corrected_text,
            original_nadi9_text=action.original_nadi9_text,
            final_status=final_st,
            created_at=action.created_at,
        )
        self.session.add(model)
        self.session.commit()
        return model

    def list_review_items_for_run(
        self, run_id: str, pending_only: bool = False
    ) -> list[ReviewItemModel]:
        stmt = select(ReviewItemModel).where(ReviewItemModel.run_id == run_id)
        if pending_only:
            stmt = stmt.where(ReviewItemModel.resolved == False)  # noqa: E712
        return list(self.session.scalars(stmt).all())

    def list_actions_for_run(self, run_id: str) -> list[HumanReviewActionModel]:
        return list(
            self.session.scalars(
                select(HumanReviewActionModel).where(
                    HumanReviewActionModel.run_id == run_id
                )
            ).all()
        )


class AuditRepository:
    """Repository for immutable AuditEvent persistence."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def log_event(
        self,
        event_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        subtitle_id: str | None = None,
    ) -> AuditEventModel:
        model = AuditEventModel(
            event_id=event_id,
            run_id=run_id,
            subtitle_id=subtitle_id,
            event_type=event_type,
            payload_json=payload,
            created_at=datetime.now(UTC),
        )
        self.session.add(model)
        self.session.commit()
        return model

    def list_events_for_run(self, run_id: str) -> list[AuditEventModel]:
        return list(
            self.session.scalars(
                select(AuditEventModel)
                .where(AuditEventModel.run_id == run_id)
                .order_by(AuditEventModel.created_at.asc())
            ).all()
        )
