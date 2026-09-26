import json

from typer.testing import CliRunner

from nadi9.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "NADI-9" in result.output or "Usage" in result.output


def test_cli_run_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--episodes" in result.output
    assert "--evidence" in result.output
    assert "--output" in result.output


def test_cli_run_sample_data(tmp_path):
    output_dir = tmp_path / "run-001"

    result = runner.invoke(
        app,
        [
            "run",
            "--episodes",
            "data/episodes/sample_episode.jsonl",
            "--evidence",
            "data/evidence/sample_evidence.jsonl",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "NADI9 Episode Processing" in result.output
    assert "Run completed." in result.output

    report_file = output_dir / "report.json"
    decisions_file = output_dir / "decisions.json"
    review_items_file = output_dir / "review_items.json"

    assert report_file.exists()
    assert decisions_file.exists()
    assert review_items_file.exists()

    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    decisions_data = json.loads(decisions_file.read_text(encoding="utf-8"))
    review_items_data = json.loads(review_items_file.read_text(encoding="utf-8"))

    assert report_data["total_subtitles_processed"] == 3
    assert report_data["accepted_count"] == 1
    assert report_data["human_review_count"] == 2

    assert len(decisions_data) == 3
    assert decisions_data[0]["subtitle_id"] == "EP001-L001"
    assert decisions_data[0]["status"] == "accepted"

    assert len(review_items_data) == 2


def test_cli_run_overwrite_protection(tmp_path):
    output_dir = tmp_path / "run-001"
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = output_dir / "existing.txt"
    dummy_file.write_text("existing content", encoding="utf-8")

    # Should fail without --force
    result = runner.invoke(
        app,
        [
            "run",
            "--episodes",
            "data/episodes/sample_episode.jsonl",
            "--evidence",
            "data/evidence/sample_evidence.jsonl",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "exists and is not empty" in result.output

    # Should succeed with --force
    result_force = runner.invoke(
        app,
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

    assert result_force.exit_code == 0


def test_cli_run_invalid_paths(tmp_path):
    output_dir = tmp_path / "run-invalid"

    result = runner.invoke(
        app,
        [
            "run",
            "--episodes",
            "non_existent_file.jsonl",
            "--evidence",
            "data/evidence/sample_evidence.jsonl",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert not (output_dir / "report.json").exists()


def test_cli_runs_list_and_show(tmp_path):
    db_file = tmp_path / "test_cli_runs.db"
    db_url = f"sqlite:///{db_file.resolve()}"

    from nadi9.domain.enums import ConfidenceLevel, DecisionStatus
    from nadi9.domain.models import SubtitleDecision
    from nadi9.storage.database import create_db_engine, create_session_factory, init_db
    from nadi9.storage.repositories import DecisionRepository, ReviewRepository, RunRepository

    engine = create_db_engine(db_url)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        run_repo = RunRepository(session)
        dec_repo = DecisionRepository(session)
        rev_repo = ReviewRepository(session)

        run_repo.create_run(run_id="RUN-CLI-001", episode_id="EP-CLI-001")
        dec = SubtitleDecision(
            subtitle_id="EP-CLI-001-L001",
            source_text="Hello",
            nadi9_text="Namaskar",
            status=DecisionStatus.ACCEPTED,
            confidence=ConfidenceLevel.HIGH,
            confidence_reason="Verified candidate",
            evidence_ids=["E-1"],
            hypothesis_ids=["H-1"],
            conflicts=[],
            human_review_required=False,
            review_question=None,
        )
        dec_repo.save_decision("RUN-CLI-001", dec)
        run_repo.update_run_status(
            "RUN-CLI-001",
            "COMPLETED",
            processing_counts={"total": 1, "processed": 1, "accepted": 1, "review": 0},
        )

    # Test runs list
    list_res = runner.invoke(app, ["runs", "list", "--db", str(db_file)])
    assert list_res.exit_code == 0
    assert "RUN-CLI-001" in list_res.output
    assert "EP-CLI-001" in list_res.output

    # Test runs show
    show_res = runner.invoke(app, ["runs", "show", "RUN-CLI-001", "--db", str(db_file)])
    assert show_res.exit_code == 0
    assert "Run Details: RUN-CLI-001" in show_res.output
    assert "EP-CLI-001-L001" in show_res.output

    # Test runs show non-existent
    show_missing = runner.invoke(app, ["runs", "show", "RUN-MISSING", "--db", str(db_file)])
    assert show_missing.exit_code != 0
    assert "not found" in show_missing.output
