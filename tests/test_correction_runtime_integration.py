import json
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from nadi9.api.app import app
from nadi9.api.dependencies import get_db
from nadi9.cli import app as cli_app
from nadi9.storage import create_db_engine, create_session_factory, init_db

cli_runner = CliRunner()
api_client = TestClient(app)


def test_cli_correction_triggers_targeted_replanning(tmp_path):
    """Integration test verifying CLI `review correct --affected` triggers apply_rule_correction."""
    output_dir = tmp_path / "cli-correction-run"

    # Step 1: Run episode processing via CLI to generate artifacts
    run_res = cli_runner.invoke(
        cli_app,
        [
            "run",
            "--episodes",
            "data/episodes/sample_episode.jsonl",
            "--evidence",
            "data/evidence/sample_evidence.jsonl",
            "--output",
            str(output_dir),
            "--force",
        ],
    )
    assert run_res.exit_code == 0

    review_items_file = output_dir / "review_items.json"
    decisions_file = output_dir / "decisions.json"
    learned_rules_file = output_dir / "learned_rules.json"

    assert review_items_file.exists()
    reviews_data = json.loads(review_items_file.read_text(encoding="utf-8"))
    assert len(reviews_data) >= 1
    target_review_id = reviews_data[0]["review_id"]
    target_sub_id = reviews_data[0]["subtitle_id"]

    # Initial snapshot of decisions
    initial_decisions = {d["subtitle_id"]: d["nadi9_text"] for d in json.loads(decisions_file.read_text(encoding="utf-8"))}

    # Step 2: Execute CLI correction affecting target_sub_id and EP001-L003
    corr_res = cli_runner.invoke(
        cli_app,
        [
            "review",
            "correct",
            "--input",
            str(output_dir),
            "--review-id",
            target_review_id,
            "--text",
            "Human corrected honorific text",
            "--affected",
            f"{target_sub_id},EP001-L003",
        ],
    )
    assert corr_res.exit_code == 0
    assert "corrected successfully" in corr_res.output

    # Step 3: Verify selective replanning & persistence
    updated_decisions = json.loads(decisions_file.read_text(encoding="utf-8"))
    dec_by_id = {d["subtitle_id"]: d for d in updated_decisions}

    # Target subtitle accepted with human text
    assert dec_by_id[target_sub_id]["nadi9_text"] == "Human corrected honorific text"
    assert dec_by_id[target_sub_id]["status"] == "accepted"

    # Unaffected EP001-L001 remains unchanged
    assert dec_by_id["EP001-L001"]["nadi9_text"] == initial_decisions["EP001-L001"]

    # Learned rules updated with affected_subtitle_ids
    assert learned_rules_file.exists()
    rules_data = json.loads(learned_rules_file.read_text(encoding="utf-8"))
    assert len(rules_data) >= 1
    latest_rule = rules_data[-1]
    assert target_sub_id in latest_rule["affected_subtitle_ids"]
    assert "EP001-L003" in latest_rule["affected_subtitle_ids"]


def test_api_correction_triggers_targeted_replanning(tmp_path):
    """Integration test verifying API review correction with affected_subtitle_ids triggers apply_rule_correction."""
    output_dir = tmp_path / "api-correction-run"

    # Step 1: Run episode processing to generate artifacts
    run_res = cli_runner.invoke(
        cli_app,
        [
            "run",
            "--episodes",
            "data/episodes/sample_episode.jsonl",
            "--evidence",
            "data/evidence/sample_evidence.jsonl",
            "--output",
            str(output_dir),
            "--force",
        ],
    )
    assert run_res.exit_code == 0

    review_items_file = output_dir / "review_items.json"
    reviews_data = json.loads(review_items_file.read_text(encoding="utf-8"))
    assert len(reviews_data) >= 1
    target_review_id = reviews_data[0]["review_id"]
    target_sub_id = reviews_data[0]["subtitle_id"]

    # Step 2: Invoke API POST /reviews/{review_id}/correct with affected_subtitle_ids
    corr_payload = {
        "actor": "api_reviewer",
        "reason": "Rule update via API",
        "text": "API Respected Text",
        "affected_subtitle_ids": [target_sub_id, "EP001-L003"],
    }

    corr_resp = api_client.post(
        f"/reviews/{target_review_id}/correct?run_dir={output_dir}",
        json=corr_payload,
    )
    assert corr_resp.status_code == 200
    action_data = corr_resp.json()
    assert action_data["action_type"] == "correct"
    assert action_data["corrected_text"] == "API Respected Text"

    # Step 3: Verify artifact persistence & targeted replanning
    decisions_file = output_dir / "decisions.json"
    learned_rules_file = output_dir / "learned_rules.json"

    updated_decisions = json.loads(decisions_file.read_text(encoding="utf-8"))
    dec_by_id = {d["subtitle_id"]: d for d in updated_decisions}

    assert dec_by_id[target_sub_id]["nadi9_text"] == "API Respected Text"
    assert dec_by_id[target_sub_id]["status"] == "accepted"

    assert learned_rules_file.exists()
    rules_data = json.loads(learned_rules_file.read_text(encoding="utf-8"))
    assert len(rules_data) >= 1
    assert "EP001-L003" in rules_data[-1]["affected_subtitle_ids"]
