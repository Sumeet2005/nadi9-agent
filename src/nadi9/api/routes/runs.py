from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from nadi9.api.dependencies import get_db
from nadi9.api.schemas import ProcessRunRequest, ReviewActionApiRequest, RunSummaryResponse
from nadi9.domain.models import EpisodeLine, EvidenceRecord
from nadi9.processor import EpisodeProcessor
from nadi9.providers.mock import MockLLMProvider
from nadi9.storage.repositories import DecisionRepository, ReviewRepository, RunRepository

router = APIRouter(prefix="/runs", tags=["Runs"])


def default_mock_provider() -> MockLLMProvider:
    return MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "greeting",
                "statement": "Namaskar.",
                "supporting_evidence": ["E-001"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "high",
            }
        ]
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def create_run(
    req: ProcessRunRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Execute end-to-end subtitle processing over inline records or file paths."""

    try:
        processor = EpisodeProcessor(
            provider=default_mock_provider(),
            db_session=db,
        )

        if req.episode_file_path and req.evidence_file_path:
            report = processor.process_files(
                episode_path=req.episode_file_path,
                evidence_path=req.evidence_file_path,
                run_id=req.run_id,
                episode_id=req.episode_id,
            )
        elif req.episode_lines and req.evidence_records:
            lines = [EpisodeLine.model_validate(l) for l in req.episode_lines]
            evidence = [EvidenceRecord.model_validate(e) for e in req.evidence_records]
            report = processor.process_records(
                episode_lines=lines,
                evidence=evidence,
                run_id=req.run_id,
                episode_id=req.episode_id,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either episode/evidence JSONL file paths or inline records.",
            )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process run: {exc}",
        ) from exc


@router.get("", response_model=list[dict[str, Any]])
def list_runs(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    repo = RunRepository(db)
    runs = repo.list_runs()
    return [
        {
            "run_id": r.run_id,
            "episode_id": r.episode_id,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "processing_counts": r.processing_counts,
        }
        for r in runs
    ]


@router.get("/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    repo = RunRepository(db)
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )
    return {
        "run_id": run.run_id,
        "episode_id": run.episode_id,
        "status": run.status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "budget_info": run.budget_info,
        "processing_counts": run.processing_counts,
        "error_info": run.error_info,
    }


@router.get("/{run_id}/decisions")
def get_run_decisions(run_id: str, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    repo = DecisionRepository(db)
    decisions = repo.list_decisions_for_run(run_id)
    return [
        {
            "subtitle_id": d.subtitle_id,
            "source_text": d.source_text,
            "nadi9_text": d.nadi9_text,
            "status": d.status,
            "confidence": d.confidence,
            "confidence_reason": d.confidence_reason,
            "human_review_required": d.human_review_required,
        }
        for d in decisions
    ]


@router.get("/{run_id}/reviews")
def get_run_reviews(run_id: str, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    repo = ReviewRepository(db)
    items = repo.list_review_items_for_run(run_id)
    return [
        {
            "review_id": r.review_id,
            "subtitle_id": r.subtitle_id,
            "reason": r.reason,
            "question": r.question,
            "priority": r.priority,
            "resolved": r.resolved,
        }
        for r in items
    ]


def _find_run_review_item(run_id: str, review_id: str, db: Session) -> Any:
    run_repo = RunRepository(db)
    if not run_repo.get_run(run_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    rev_repo = ReviewRepository(db)
    items = rev_repo.list_review_items_for_run(run_id)
    target = next((i for i in items if i.review_id == review_id), None)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review '{review_id}' for run '{run_id}' not found.",
        )
    if target.resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Review '{review_id}' has already been resolved.",
        )
    return target


@router.post("/{run_id}/reviews/{review_id}/approve")
def approve_run_review(
    run_id: str,
    review_id: str,
    req: ReviewActionApiRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Approve a pending review item for a run and resume workflow execution."""
    item = _find_run_review_item(run_id, review_id, db)
    processor = EpisodeProcessor(provider=default_mock_provider(), db_session=db)
    try:
        final_decision = processor.resume_human_review(
            run_id=run_id,
            subtitle_id=item.subtitle_id,
            action_type="approve",
            actor=req.actor,
        )
        return final_decision.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to resume workflow review: {exc}",
        ) from exc


@router.post("/{run_id}/reviews/{review_id}/reject")
def reject_run_review(
    run_id: str,
    review_id: str,
    req: ReviewActionApiRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Reject a pending review item for a run and resume workflow execution."""
    if not req.reason or not req.reason.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejection reason is required.",
        )
    item = _find_run_review_item(run_id, review_id, db)
    processor = EpisodeProcessor(provider=default_mock_provider(), db_session=db)
    try:
        final_decision = processor.resume_human_review(
            run_id=run_id,
            subtitle_id=item.subtitle_id,
            action_type="reject",
            reason=req.reason.strip(),
            actor=req.actor,
        )
        return final_decision.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to resume workflow review: {exc}",
        ) from exc


@router.post("/{run_id}/reviews/{review_id}/correct")
def correct_run_review(
    run_id: str,
    review_id: str,
    req: ReviewActionApiRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Supply human correction for a review item and resume workflow execution."""
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrected subtitle text is required.",
        )
    item = _find_run_review_item(run_id, review_id, db)
    processor = EpisodeProcessor(provider=default_mock_provider(), db_session=db)
    try:
        final_decision = processor.resume_human_review(
            run_id=run_id,
            subtitle_id=item.subtitle_id,
            action_type="correct",
            text=req.text.strip(),
            reason=req.reason,
            actor=req.actor,
        )
        return final_decision.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to resume workflow review: {exc}",
        ) from exc


@router.post("/{run_id}/reviews/{review_id}/resume")
def resume_run_review(
    run_id: str,
    review_id: str,
    req: ReviewActionApiRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Generic endpoint to resume a paused review thread using action_type."""
    action_type = req.action_type or "approve"
    if action_type == "reject" and (not req.reason or not req.reason.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejection reason is required.",
        )
    if action_type == "correct" and (not req.text or not req.text.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrected subtitle text is required.",
        )

    item = _find_run_review_item(run_id, review_id, db)
    processor = EpisodeProcessor(provider=default_mock_provider(), db_session=db)
    try:
        final_decision = processor.resume_human_review(
            run_id=run_id,
            subtitle_id=item.subtitle_id,
            action_type=action_type,
            reason=req.reason,
            text=req.text,
            actor=req.actor,
        )
        return final_decision.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to resume workflow review: {exc}",
        ) from exc
