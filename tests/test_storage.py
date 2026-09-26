import pytest
from sqlalchemy.orm import Session

from nadi9.domain.enums import ConfidenceLevel, DecisionStatus, ReviewPriority, VerificationStatus
from nadi9.domain.models import (
    HumanReviewAction,
    ReviewItem,
    SubtitleDecision,
    VerificationResult,
)
from nadi9.storage import (
    AuditRepository,
    DecisionRepository,
    ReviewRepository,
    RunRepository,
    create_db_engine,
    create_session_factory,
    init_db,
)


@pytest.fixture
def db_session(tmp_path):
    db_file = tmp_path / "test_nadi9.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_db_engine(db_url)
    init_db(engine)
    session_factory = create_session_factory(engine)

    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_run_repository_crud(db_session: Session):
    repo = RunRepository(db_session)

    run = repo.create_run(run_id="run-001", episode_id="ep-001", config_snapshot={"env": "test"})
    assert run.run_id == "run-001"
    assert run.status == "CREATED"

    updated = repo.update_run_status("run-001", "RUNNING")
    assert updated is not None
    assert updated.status == "RUNNING"
    assert updated.started_at is not None

    completed = repo.update_run_status(
        "run-001", "COMPLETED", processing_counts={"subtitles": 5}
    )
    assert completed is not None
    assert completed.status == "COMPLETED"
    assert completed.completed_at is not None

    fetched = repo.get_run("run-001")
    assert fetched is not None
    assert fetched.processing_counts["subtitles"] == 5

    runs = repo.list_runs()
    assert len(runs) == 1


def test_decision_repository(db_session: Session):
    run_repo = RunRepository(db_session)
    run_repo.create_run("run-002", "ep-002")

    dec_repo = DecisionRepository(db_session)
    decision = SubtitleDecision(
        subtitle_id="SUB-001",
        source_text="Hello",
        nadi9_text="Namaskar",
        status=DecisionStatus.ACCEPTED,
        evidence_ids=["E-001"],
        hypothesis_ids=["H-001"],
        confidence=ConfidenceLevel.HIGH,
        confidence_reason="Verified",
        verification=VerificationResult(
            status=VerificationStatus.PASSED,
            checks={"check1": True},
        ),
    )

    model = dec_repo.save_decision("run-002", decision)
    assert model.subtitle_id == "SUB-001"
    assert model.nadi9_text == "Namaskar"
    assert model.status == "accepted"

    decisions = dec_repo.list_decisions_for_run("run-002")
    assert len(decisions) == 1
    assert decisions[0].nadi9_text == "Namaskar"


def test_review_repository(db_session: Session):
    run_repo = RunRepository(db_session)
    run_repo.create_run("run-003", "ep-003")

    rev_repo = ReviewRepository(db_session)

    item = ReviewItem(
        review_id="REV-001",
        subtitle_id="SUB-001",
        reason="Conflict",
        question="Review evidence",
        evidence_ids=["E-001", "E-002"],
        priority=ReviewPriority.HIGH,
    )

    item_model = rev_repo.save_review_item("run-003", item)
    assert item_model.review_id == "REV-001"
    assert item_model.resolved is False

    action = HumanReviewAction(
        action_id="ACT-001",
        review_id="REV-001",
        subtitle_id="SUB-001",
        action_type="approve",
        original_nadi9_text="Namaskar",
        final_status=DecisionStatus.ACCEPTED,
    )

    act_model = rev_repo.save_review_action("run-003", action)
    assert act_model.action_id == "ACT-001"
    assert act_model.action_type == "approve"

    pending = rev_repo.list_review_items_for_run("run-003", pending_only=True)
    assert len(pending) == 1

    item.resolved = True
    rev_repo.save_review_item("run-003", item)

    pending_after = rev_repo.list_review_items_for_run("run-003", pending_only=True)
    assert len(pending_after) == 0


def test_audit_repository(db_session: Session):
    run_repo = RunRepository(db_session)
    run_repo.create_run("run-004", "ep-004")

    audit_repo = AuditRepository(db_session)
    event = audit_repo.log_event(
        event_id="EV-001",
        run_id="run-004",
        event_type="run_started",
        payload={"message": "Started"},
        subtitle_id="SUB-001",
    )

    assert event.event_id == "EV-001"
    assert event.event_type == "run_started"

    events = audit_repo.list_events_for_run("run-004")
    assert len(events) == 1
    assert events[0].payload_json["message"] == "Started"
