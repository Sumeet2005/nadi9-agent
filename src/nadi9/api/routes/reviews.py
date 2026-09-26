from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from nadi9.api.dependencies import get_db
from nadi9.api.schemas import ReviewActionApiRequest
from nadi9.audit.review_service import (
    ReviewAlreadyResolvedError,
    ReviewManager,
    ReviewNotFoundError,
    RunArtifactError,
)

router = APIRouter(prefix="/reviews", tags=["Reviews"])


@router.post("/{review_id}/approve")
def approve_review_endpoint(
    review_id: str,
    req: ReviewActionApiRequest,
    run_dir: str = "outputs/run-001",
) -> dict[str, Any]:
    """Approve a pending review item."""
    try:
        manager = ReviewManager(run_dir)
        action = manager.approve(review_id, actor=req.actor)
        return action.model_dump(mode="json")
    except ReviewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ReviewAlreadyResolvedError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RunArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post("/{review_id}/reject")
def reject_review_endpoint(
    review_id: str,
    req: ReviewActionApiRequest,
    run_dir: str = "outputs/run-001",
) -> dict[str, Any]:
    """Reject a pending review item."""
    if not req.reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejection reason is required.",
        )
    try:
        manager = ReviewManager(run_dir)
        action = manager.reject(review_id, reason=req.reason, actor=req.actor)
        return action.model_dump(mode="json")
    except ReviewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ReviewAlreadyResolvedError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/{review_id}/correct")
def correct_review_endpoint(
    review_id: str,
    req: ReviewActionApiRequest,
    run_dir: str = "outputs/run-001",
) -> dict[str, Any]:
    """Supply human correction for a review item."""
    if not req.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrected subtitle text is required.",
        )
    try:
        manager = ReviewManager(run_dir)
        action = manager.correct(
            review_id, text=req.text, reason=req.reason, actor=req.actor
        )
        return action.model_dump(mode="json")
    except ReviewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ReviewAlreadyResolvedError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
