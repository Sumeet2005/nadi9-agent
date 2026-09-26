import pytest
from typer.testing import CliRunner

from nadi9.cli import app
from nadi9.domain.errors import BudgetExceededError

runner = CliRunner()


def test_cli_run_enforces_budget_and_fails(tmp_path, monkeypatch):
    """Integration test verifying CLI run_command enforces model call budget."""
    output_dir = tmp_path / "run-budget-test"

    # Set NADI9_MAX_MODEL_CALLS to 1 (too low for 3 subtitle lines requiring LLM calls)
    monkeypatch.setenv("NADI9_MAX_MODEL_CALLS", "1")

    res = runner.invoke(
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

    # CLI should catch exception and exit with code 1, echoing budget error message
    assert res.exit_code == 1
    assert "Model-call budget exceeded" in res.output or "Error processing episodes" in res.output
