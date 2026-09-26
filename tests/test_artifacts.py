import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from nadi9.artifacts.exporter import export_sample_run_artifacts, format_srt_timestamp
from nadi9.audit.report import AuditReport, DecisionAuditDetail
from nadi9.cli import app
from nadi9.domain.enums import ConfidenceLevel, DecisionStatus, VerificationStatus
from nadi9.domain.models import EpisodeLine, LearnedRule, ReviewItem, VerificationResult

runner = CliRunner()


def test_srt_timestamp_formatting():
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(3.5) == "00:00:03,500"
    assert format_srt_timestamp(65.25) == "00:01:05,250"
    assert format_srt_timestamp(3665.123) == "01:01:05,123"


def test_artifact_exporter_generates_all_files(tmp_path):
    output_dir = tmp_path / "sample_run"

    dec1 = DecisionAuditDetail(
        subtitle_id="SUB-001",
        source_text="Hello",
        nadi9_text="Namaskar.",
        status=DecisionStatus.ACCEPTED,
        confidence=ConfidenceLevel.HIGH,
        confidence_reason="Verified",
        human_review_required=False,
    )
    dec2 = DecisionAuditDetail(
        subtitle_id="SUB-002",
        source_text="Rare phrase",
        nadi9_text="Translation unavailable: insufficient supporting evidence.",
        status=DecisionStatus.REVIEW_REQUIRED,
        confidence=ConfidenceLevel.LOW,
        confidence_reason="Insufficient evidence",
        human_review_required=True,
    )

    review1 = ReviewItem(
        review_id="REV-1",
        subtitle_id="SUB-002",
        reason="No evidence",
        question="Please confirm translation",
        resolved=False,
    )

    report = AuditReport(
        run_id="run-test",
        episode_id="ep-test",
        total_subtitles_processed=2,
        accepted_count=1,
        human_review_count=1,
        decisions=[dec1, dec2],
        review_items=[review1],
    )

    rule1 = LearnedRule(
        rule_id="RULE-1",
        category="greeting",
        statement="Use Namaskar for formal greeting",
        affected_subtitle_ids=["SUB-001"],
    )

    lines = [
        EpisodeLine(subtitle_id="SUB-001", source_text="Hello", start_time=1.0, end_time=3.0),
        EpisodeLine(subtitle_id="SUB-002", source_text="Rare phrase", start_time=3.5, end_time=5.5),
    ]

    paths = export_sample_run_artifacts(
        output_dir=output_dir,
        report=report,
        episode_lines=lines,
        learned_rules=[rule1],
    )

    # Check file existence
    assert paths["subtitles_srt"].exists()
    assert paths["subtitle_decisions_jsonl"].exists()
    assert paths["learned_rules_json"].exists()
    assert paths["review_queue_json"].exists()
    assert paths["final_report_md"].exists()

    # 1. subtitles.srt check
    srt_text = paths["subtitles_srt"].read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:03,000" in srt_text
    assert "Namaskar." in srt_text
    assert "Translation unavailable: insufficient supporting evidence." in srt_text

    # 2. subtitle_decisions.jsonl check
    jsonl_lines = paths["subtitle_decisions_jsonl"].read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 2
    rec1 = json.loads(jsonl_lines[0])
    assert rec1["subtitle_id"] == "SUB-001"
    assert rec1["nadi9_text"] == "Namaskar."

    # 3. learned_rules.json check
    rules_json = json.loads(paths["learned_rules_json"].read_text(encoding="utf-8"))
    assert len(rules_json) == 1
    assert rules_json[0]["rule_id"] == "RULE-1"

    # 4. review_queue.json check
    queue_json = json.loads(paths["review_queue_json"].read_text(encoding="utf-8"))
    assert len(queue_json) == 1
    assert queue_json[0]["review_id"] == "REV-1"

    # 5. final_report.md check
    md_text = paths["final_report_md"].read_text(encoding="utf-8")
    assert "# NADI-9 Final Report" in md_text
    assert "run-test" in md_text
    assert "RULE-1" in md_text
    assert "## Final Release Recommendation" in md_text
    assert "HOLD - HUMAN REVIEW REQUIRED" in md_text


def test_artifact_exporter_handles_empty_rules_and_reviews(tmp_path):
    output_dir = tmp_path / "sample_run_empty"
    dec = DecisionAuditDetail(
        subtitle_id="SUB-001",
        source_text="Hello",
        nadi9_text="Hello",
        status=DecisionStatus.ACCEPTED,
        confidence=ConfidenceLevel.HIGH,
        confidence_reason="Verified",
        human_review_required=False,
    )
    report = AuditReport(
        run_id="run-empty",
        episode_id="ep-empty",
        total_subtitles_processed=1,
        accepted_count=1,
        human_review_count=0,
        decisions=[dec],
        review_items=[],
    )

    paths = export_sample_run_artifacts(
        output_dir=output_dir,
        report=report,
        learned_rules=[],
    )

    rules_json = json.loads(paths["learned_rules_json"].read_text(encoding="utf-8"))
    assert rules_json == []

    queue_json = json.loads(paths["review_queue_json"].read_text(encoding="utf-8"))
    assert queue_json == []

    md_text = paths["final_report_md"].read_text(encoding="utf-8")
    assert "None generated during this run." in md_text
    assert "## Final Release Recommendation" in md_text
    assert "APPROVED FOR RELEASE" in md_text


def test_cli_run_generates_all_artifacts(tmp_path):
    output_dir = tmp_path / "sample_run_cli"
    episodes_file = Path("data/episodes/sample_episode.jsonl")
    evidence_file = Path("data/evidence/sample_evidence.jsonl")

    res = runner.invoke(
        app,
        [
            "run",
            "--episodes",
            str(episodes_file),
            "--evidence",
            str(evidence_file),
            "--output",
            str(output_dir),
            "--force",
        ],
    )
    assert res.exit_code == 0

    # Backward compatible files
    assert (output_dir / "report.json").exists()
    assert (output_dir / "decisions.json").exists()
    assert (output_dir / "review_items.json").exists()

    # Required Phase 8 assignment package files
    assert (output_dir / "subtitles.srt").exists()
    assert (output_dir / "subtitle_decisions.jsonl").exists()
    assert (output_dir / "learned_rules.json").exists()
    assert (output_dir / "review_queue.json").exists()
    assert (output_dir / "final_report.md").exists()
    
    final_report = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert "## Final Release Recommendation" in final_report
