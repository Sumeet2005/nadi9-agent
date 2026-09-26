import json
from pathlib import Path
from typing import Any

import typer

from nadi9.audit.review_service import ReviewError, ReviewManager
from nadi9.budget import BudgetManager
from nadi9.config import get_settings
from nadi9.processor import EpisodeProcessor
from nadi9.providers.budgeted import BudgetedLLMProvider
from nadi9.providers.mock import MockLLMProvider
from nadi9.providers.openai import OpenAIProvider
from nadi9.storage.database import create_db_engine, create_session_factory, init_db

app = typer.Typer(
    name="nadi9",
    help="NADI-9 Evidence-Grounded Agentic Subtitle Decision System",
    add_completion=False,
)

review_app = typer.Typer(
    name="review",
    help="Human-in-the-Loop review management commands.",
    add_completion=False,
)
app.add_typer(review_app, name="review")

runs_app = typer.Typer(
    name="runs",
    help="Durable execution run management commands.",
    add_completion=False,
)
app.add_typer(runs_app, name="runs")


@app.callback()
def main_callback() -> None:
    """NADI-9 CLI Entry Point."""


def get_sample_mock_responses() -> list[dict[str, Any]]:
    """Return default mock responses for sample episode processing."""
    return [
        {
            "hypothesis_id": "H-001",
            "category": "greeting",
            "statement": "Namaskar.",
            "supporting_evidence": ["E-001"],
            "counterexamples": [],
            "status": "proposed",
            "confidence": "high",
        },
        {
            "hypothesis_id": "H-002",
            "category": "farewell",
            "statement": "Alvida.",
            "supporting_evidence": ["E-002", "E-003"],
            "counterexamples": [],
            "status": "proposed",
            "confidence": "medium",
        },
        {
            "hypothesis_id": "H-003",
            "category": "idiom",
            "statement": "Archaic translation.",
            "supporting_evidence": ["E-999"],
            "counterexamples": [],
            "status": "proposed",
            "confidence": "low",
        },
    ]


@app.command(name="run")
def run_command(
    episodes: Path = typer.Option(
        ...,
        "--episodes",
        "-e",
        help="Path to input episodes JSONL file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    evidence: Path = typer.Option(
        ...,
        "--evidence",
        "-v",
        help="Path to input evidence JSONL file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Directory to save output JSON artifacts.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing output directory if it exists.",
    ),
    provider_name: str = typer.Option(
        "mock",
        "--provider",
        "-p",
        help="LLM provider implementation to use (mock or openai, default: mock).",
    ),
) -> None:
    """Run end-to-end subtitle decision processing over episode lines and evidence."""

    if output.exists() and any(output.iterdir()) and not force:
        typer.echo(
            f"Error: Output directory '{output}' exists and is not empty. "
            "Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)

    typer.echo("NADI9 Episode Processing")
    typer.echo("------------------------")
    typer.echo(f"Episodes file: {episodes}")
    typer.echo(f"Evidence file: {evidence}")
    typer.echo("")
    typer.echo("Processing...")

    try:
        settings = get_settings()
        p_name = provider_name.lower()
        if p_name == "mock":
            llm_provider = MockLLMProvider(responses=get_sample_mock_responses())
        elif p_name == "openai":
            llm_provider = OpenAIProvider()
        else:
            typer.echo(
                f"Error: Provider '{provider_name}' is not supported.",
                err=True,
            )
            raise typer.Exit(code=1)

        llm_provider = BudgetedLLMProvider(
            llm_provider,
            BudgetManager(
                max_model_calls=settings.max_model_calls,
                max_tool_calls=settings.max_tool_calls,
            ),
        )

        engine = create_db_engine(settings.database_url)
        init_db(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            processor = EpisodeProcessor(
                provider=llm_provider,
                max_model_calls=settings.max_model_calls,
                max_tool_calls=settings.max_tool_calls,
                db_session=session,
            )

            report = processor.process_files(
                episode_path=episodes,
                evidence_path=evidence,
            )

        if any("budget" in str(err).lower() or "exceeded" in str(err).lower() for err in report.errors):
            typer.echo(f"Error processing episodes: {report.errors[0]}", err=True)
            raise typer.Exit(code=1)

        for decision in report.decisions:
            status_val = (
                decision.status.value
                if hasattr(decision.status, "value")
                else str(decision.status)
            )
            status_tag = (
                "[ACCEPTED]"
                if status_val == "accepted"
                else "[REVIEW REQUIRED]"
            )
            typer.echo(f"[{decision.subtitle_id}] {status_tag}")

        typer.echo("")
        typer.echo("Run completed.")
        typer.echo(f"Total processed: {report.total_subtitles_processed}")
        typer.echo(f"Accepted: {report.accepted_count}")
        typer.echo(f"Review required: {report.human_review_count}")

        # Save output JSON artifacts (backward compatibility)
        report_file = output / "report.json"
        decisions_file = output / "decisions.json"
        review_items_file = output / "review_items.json"

        report_file.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

        decisions_data = [d.model_dump(mode="json") for d in report.decisions]
        decisions_file.write_text(
            json.dumps(decisions_data, indent=2), encoding="utf-8"
        )

        review_items_data = [r.model_dump(mode="json") for r in report.review_items]
        review_items_file.write_text(
            json.dumps(review_items_data, indent=2), encoding="utf-8"
        )

        # Generate Phase 8 required sample run package
        from nadi9.artifacts.exporter import export_sample_run_artifacts
        from nadi9.ingestion.jsonl import ingest_episode_lines
        try:
            episode_lines_list = ingest_episode_lines(episodes)
        except Exception:
            episode_lines_list = None

        export_sample_run_artifacts(
            output_dir=output,
            report=report,
            episode_lines=episode_lines_list,
            learned_rules=[],
        )

        typer.echo("")
        typer.echo(f"Audit report saved to: {report_file}")
        typer.echo(f"Decisions saved to: {decisions_file}")
        typer.echo(f"Review items saved to: {review_items_file}")
        typer.echo(f"Sample run package exported to: {output}")

    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Error processing episodes: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@review_app.command(name="list")
def review_list(
    input_dir: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to output run directory containing artifacts.",
    ),
    all_items: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Include already resolved review items.",
    ),
) -> None:
    """List pending human review items for a run."""

    try:
        manager = ReviewManager(input_dir)
        reviews = manager.list_reviews(include_resolved=all_items)

        if not reviews:
            typer.echo(f"No pending review items found in '{input_dir}'.")
            return

        typer.echo(f"Pending Reviews for run at '{input_dir}':")
        typer.echo("----------------------------------------")
        for item in reviews:
            typer.echo(f"Review ID : {item['review_id']}")
            typer.echo(f"Subtitle  : {item['subtitle_id']}")
            typer.echo(f"Priority  : {item['priority']}")
            typer.echo(f"Question  : {item['question']}")
            typer.echo(f"Reason    : {item['reason']}")
            typer.echo(f"AI Candidate Text: {item['current_nadi9_text']}")
            typer.echo("----------------------------------------")

    except ReviewError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Error listing reviews: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@review_app.command(name="approve")
def review_approve(
    input_dir: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to output run directory containing artifacts.",
    ),
    review_id: str = typer.Option(
        ...,
        "--review-id",
        "-r",
        help="Review ID to approve.",
    ),
    actor: str = typer.Option(
        "human_reviewer",
        "--actor",
        help="Identifier of human reviewer.",
    ),
) -> None:
    """Approve an AI candidate subtitle decision."""

    try:
        manager = ReviewManager(input_dir)
        action = manager.approve(review_id=review_id, actor=actor)

        typer.echo(f"Review '{review_id}' approved successfully.")
        typer.echo(f"Subtitle ID: {action.subtitle_id}")
        typer.echo(f"Final Status: {action.final_status.value}")
        typer.echo(f"Action ID: {action.action_id}")

    except ReviewError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Error approving review: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@review_app.command(name="reject")
def review_reject(
    input_dir: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to output run directory containing artifacts.",
    ),
    review_id: str = typer.Option(
        ...,
        "--review-id",
        "-r",
        help="Review ID to reject.",
    ),
    reason: str = typer.Option(
        ...,
        "--reason",
        "-m",
        help="Reason for rejecting the candidate decision.",
    ),
    actor: str = typer.Option(
        "human_reviewer",
        "--actor",
        help="Identifier of human reviewer.",
    ),
) -> None:
    """Reject an AI candidate subtitle decision with a mandatory reason."""

    try:
        manager = ReviewManager(input_dir)
        action = manager.reject(review_id=review_id, reason=reason, actor=actor)

        typer.echo(f"Review '{review_id}' rejected successfully.")
        typer.echo(f"Subtitle ID: {action.subtitle_id}")
        typer.echo(f"Rejection Reason: {action.reason}")
        typer.echo(f"Final Status: {action.final_status.value}")
        typer.echo(f"Action ID: {action.action_id}")

    except ReviewError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Error rejecting review: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@review_app.command(name="correct")
def review_correct(
    input_dir: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to output run directory containing artifacts.",
    ),
    review_id: str = typer.Option(
        ...,
        "--review-id",
        "-r",
        help="Review ID to correct.",
    ),
    text: str = typer.Option(
        ...,
        "--text",
        "-t",
        help="Human-corrected subtitle text.",
    ),
    reason: str = typer.Option(
        "",
        "--reason",
        "-m",
        help="Optional note explaining the correction.",
    ),
    actor: str = typer.Option(
        "human_reviewer",
        "--actor",
        help="Identifier of human reviewer.",
    ),
    affected: str | None = typer.Option(
        None,
        "--affected",
        "-a",
        help="Optional comma-separated list of additional affected subtitle IDs for targeted replanning.",
    ),
) -> None:
    """Provide human-corrected subtitle text for a review item, with optional targeted selective replanning for affected subtitles."""

    try:
        affected_ids = (
            [s.strip() for s in affected.split(",") if s.strip()]
            if affected
            else None
        )
        manager = ReviewManager(input_dir)
        action = manager.correct(
            review_id=review_id,
            text=text,
            reason=reason,
            actor=actor,
            affected_subtitle_ids=affected_ids,
        )

        typer.echo(f"Review '{review_id}' corrected successfully.")
        typer.echo(f"Subtitle ID: {action.subtitle_id}")
        typer.echo(f"Original Text : {action.original_nadi9_text}")
        typer.echo(f"Corrected Text: {action.corrected_text}")
        typer.echo(f"Final Status  : {action.final_status.value}")
        typer.echo(f"Action ID     : {action.action_id}")

    except ReviewError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Error correcting review: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@runs_app.command(name="list")
def runs_list(
    db_path: Path | None = typer.Option(
        None,
        "--db",
        help="Path to SQLite database file (default: from NADI9_DATABASE_URL settings).",
    ),
) -> None:
    """List execution runs from the SQLite database."""
    from nadi9.config import get_settings
    from nadi9.storage.database import create_db_engine, create_session_factory, init_db
    from nadi9.storage.repositories import RunRepository

    try:
        settings = get_settings()
        db_url = f"sqlite:///{db_path.resolve()}" if db_path else settings.database_url
        engine = create_db_engine(db_url)
        init_db(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            repo = RunRepository(session)
            runs = repo.list_runs()

            if not runs:
                typer.echo("No execution runs found in database.")
                return

            typer.echo("NADI9 Execution Runs:")
            typer.echo("--------------------------------------------------")
            for r in runs:
                counts = r.processing_counts or {}
                typer.echo(f"Run ID    : {r.run_id}")
                typer.echo(f"Episode   : {r.episode_id}")
                typer.echo(f"Status    : {r.status}")
                typer.echo(
                    f"Subtitles : Total={counts.get('total', 0)}, "
                    f"Accepted={counts.get('accepted', 0)}, "
                    f"Review={counts.get('review', 0)}"
                )
                typer.echo(f"Created   : {r.created_at}")
                typer.echo("--------------------------------------------------")

    except Exception as exc:
        typer.echo(f"Error listing runs: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@runs_app.command(name="show")
def runs_show(
    run_id: str = typer.Argument(
        ...,
        help="Unique identifier of the run to show.",
    ),
    db_path: Path | None = typer.Option(
        None,
        "--db",
        help="Path to SQLite database file (default: from NADI9_DATABASE_URL settings).",
    ),
) -> None:
    """Show detailed status, decisions, and review counts for a specific execution run."""
    from nadi9.config import get_settings
    from nadi9.storage.database import create_db_engine, create_session_factory, init_db
    from nadi9.storage.repositories import DecisionRepository, ReviewRepository, RunRepository

    try:
        settings = get_settings()
        db_url = f"sqlite:///{db_path.resolve()}" if db_path else settings.database_url
        engine = create_db_engine(db_url)
        init_db(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            run_repo = RunRepository(session)
            dec_repo = DecisionRepository(session)
            rev_repo = ReviewRepository(session)

            run_obj = run_repo.get_run(run_id)
            if not run_obj:
                typer.echo(f"Error: Run '{run_id}' not found.", err=True)
                raise typer.Exit(code=1)

            decisions = dec_repo.list_decisions_for_run(run_id)
            reviews = rev_repo.list_review_items_for_run(run_id)

            counts = run_obj.processing_counts or {}
            typer.echo(f"Run Details: {run_obj.run_id}")
            typer.echo("==================================================")
            typer.echo(f"Episode ID : {run_obj.episode_id}")
            typer.echo(f"Status     : {run_obj.status}")
            typer.echo(f"Total Lines: {counts.get('total', 0)}")
            typer.echo(f"Accepted   : {counts.get('accepted', 0)}")
            typer.echo(f"Review Reqd: {counts.get('review', 0)}")
            typer.echo(f"Created At : {run_obj.created_at}")
            typer.echo("--------------------------------------------------")
            typer.echo(f"Decisions ({len(decisions)}):")
            for d in decisions:
                typer.echo(f"  [{d.subtitle_id}] status={d.status} confidence={d.confidence} text=\"{d.nadi9_text}\"")
            typer.echo("--------------------------------------------------")
            typer.echo(f"Review Items ({len(reviews)}):")
            for r in reviews:
                typer.echo(f"  [{r.review_id}] subtitle={r.subtitle_id} reason=\"{r.reason}\"")

    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Error showing run: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
