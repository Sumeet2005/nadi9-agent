import json

import pytest
from typer.testing import CliRunner

from nadi9.audit.review_service import (
    ReviewAlreadyResolvedError,
    ReviewManager,
    ReviewNotFoundError,
    RunArtifactError,
)
from nadi9.cli import app
from nadi9.domain.enums import DecisionStatus
from nadi9.processor import EpisodeProcessor
from nadi9.providers.mock import MockLLMProvider

runner = CliRunner()


def make_sample_run(tmp_path):
    episode_file = tmp_path / "episodes.jsonl"
    evidence_file = tmp_path / "evidence.jsonl"
    output_dir = tmp_path / "run-test"

    episode_file.write_text(
        json.dumps(
            {
                "subtitle_id": "SUB-001",
                "source_text": "informal farewell",
                "speaker": "Speaker B",
                "scene_id": "SCENE-01",
                "start_time": 0.0,
                "end_time": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ev1 = {
        "evidence_id": "E-001",
        "source_id": "SRC-1",
        "evidence_type": "grammar",
        "content": "informal farewell",
        "metadata": {"conflicts_with": ["E-002"]},
        "provenance": {"original_file": "g.json", "extraction_method": "manual"},
    }
    ev2 = {
        "evidence_id": "E-002",
        "source_id": "SRC-2",
        "evidence_type": "grammar",
        "content": "slang farewell",
        "metadata": {"conflicts_with": ["E-001"]},
        "provenance": {"original_file": "g.json", "extraction_method": "manual"},
    }
    evidence_file.write_text(
        json.dumps(ev1) + "\n" + json.dumps(ev2) + "\n", encoding="utf-8"
    )

    mock = MockLLMProvider(
        responses=[
            {
                "hypothesis_id": "H-001",
                "category": "farewell",
                "statement": "Original AI Text.",
                "supporting_evidence": ["E-001", "E-002"],
                "counterexamples": [],
                "status": "proposed",
                "confidence": "medium",
            }
        ]
    )

    processor = EpisodeProcessor(provider=mock)
    report = processor.process_files(
        episode_path=episode_file, evidence_path=evidence_file
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    decisions_data = [d.model_dump(mode="json") for d in report.decisions]
    (output_dir / "decisions.json").write_text(
        json.dumps(decisions_data, indent=2), encoding="utf-8"
    )
    reviews_data = [r.model_dump(mode="json") for r in report.review_items]
    (output_dir / "review_items.json").write_text(
        json.dumps(reviews_data, indent=2), encoding="utf-8"
    )

    return output_dir, reviews_data[0]["review_id"]


def test_review_manager_list_reviews(tmp_path):
    output_dir, review_id = make_sample_run(tmp_path)
    manager = ReviewManager(output_dir)

    reviews = manager.list_reviews()
    assert len(reviews) == 1
    assert reviews[0]["review_id"] == review_id
    assert reviews[0]["subtitle_id"] == "SUB-001"
    assert reviews[0]["current_nadi9_text"] == "Original AI Text."


def test_review_manager_approve(tmp_path):
    output_dir, review_id = make_sample_run(tmp_path)
    manager = ReviewManager(output_dir)

    action = manager.approve(review_id, actor="reviewer_alice")
    assert action.action_type == "approve"
    assert action.original_nadi9_text == "Original AI Text."
    assert action.final_status == DecisionStatus.ACCEPTED

    # Verify pending review list is now empty
    assert len(manager.list_reviews(include_resolved=False)) == 0

    # Verify audit actions file was created
    actions_file = output_dir / "review_actions.json"
    assert actions_file.exists()
    actions = json.loads(actions_file.read_text(encoding="utf-8"))
    assert len(actions) == 1
    assert actions[0]["actor"] == "reviewer_alice"


def test_review_manager_reject(tmp_path):
    output_dir, review_id = make_sample_run(tmp_path)
    manager = ReviewManager(output_dir)

    action = manager.reject(review_id, reason="Inaccurate terminology", actor="reviewer_bob")
    assert action.action_type == "reject"
    assert action.reason == "Inaccurate terminology"
    assert action.final_status == DecisionStatus.REJECTED

    decisions = json.loads((output_dir / "decisions.json").read_text(encoding="utf-8"))
    assert decisions[0]["status"] == "rejected"


def test_review_manager_correct(tmp_path):
    output_dir, review_id = make_sample_run(tmp_path)
    manager = ReviewManager(output_dir)

    action = manager.correct(
        review_id, text="Human Corrected Subtitle Text.", reason="Stylistic fix"
    )
    assert action.action_type == "correct"
    assert action.original_nadi9_text == "Original AI Text."
    assert action.corrected_text == "Human Corrected Subtitle Text."
    assert action.final_status == DecisionStatus.ACCEPTED

    decisions = json.loads((output_dir / "decisions.json").read_text(encoding="utf-8"))
    assert decisions[0]["nadi9_text"] == "Human Corrected Subtitle Text."
    assert decisions[0]["status"] == "accepted"


def test_review_manager_unknown_review_id(tmp_path):
    output_dir, _ = make_sample_run(tmp_path)
    manager = ReviewManager(output_dir)

    with pytest.raises(ReviewNotFoundError, match="was not found"):
        manager.approve("REV-INVALID-999")


def test_review_manager_already_resolved(tmp_path):
    output_dir, review_id = make_sample_run(tmp_path)
    manager = ReviewManager(output_dir)

    manager.approve(review_id)

    with pytest.raises(ReviewAlreadyResolvedError, match="already been resolved"):
        manager.approve(review_id)


def test_review_manager_missing_artifacts(tmp_path):
    invalid_dir = tmp_path / "non_existent_run"
    with pytest.raises(RunArtifactError, match="does not exist"):
        ReviewManager(invalid_dir)

    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    with pytest.raises(RunArtifactError, match="review_items.json is missing"):
        ReviewManager(empty_dir)


def test_cli_review_commands(tmp_path):
    output_dir, review_id = make_sample_run(tmp_path)

    # Test list
    res_list = runner.invoke(app, ["review", "list", "--input", str(output_dir)])
    assert res_list.exit_code == 0
    assert review_id in res_list.output
    assert "Original AI Text." in res_list.output

    # Test approve
    res_app = runner.invoke(
        app, ["review", "approve", "--input", str(output_dir), "--review-id", review_id]
    )
    assert res_app.exit_code == 0
    assert "approved successfully" in res_app.output

    # Test list after approval
    res_list_after = runner.invoke(app, ["review", "list", "--input", str(output_dir)])
    assert res_list_after.exit_code == 0
    assert "No pending review items found" in res_list_after.output
