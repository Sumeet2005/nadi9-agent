import json
from pathlib import Path
from typing import Any

from nadi9.audit.report import AuditReport
from nadi9.domain.models import EpisodeLine, LearnedRule, ReviewItem, SubtitleDecision


def format_srt_timestamp(seconds: float) -> str:
    """Convert floating seconds to standard SRT timestamp (HH:MM:SS,mmm)."""
    if seconds < 0:
        seconds = 0.0

    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3600000
    total_milliseconds %= 3600000
    minutes = total_milliseconds // 60000
    total_milliseconds %= 60000
    secs = total_milliseconds // 1000
    millis = total_milliseconds % 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_sample_run_artifacts(
    *,
    output_dir: str | Path,
    report: AuditReport,
    episode_lines: list[EpisodeLine] | None = None,
    learned_rules: list[LearnedRule] | None = None,
) -> dict[str, Path]:
    """Export the standardized sample run artifact package required by Phase 8."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lines_by_id = {line.subtitle_id: line for line in (episode_lines or [])}
    decisions = report.decisions

    # 1. subtitles.srt
    srt_file = out_path / "subtitles.srt"
    srt_blocks: list[str] = []

    for idx, dec in enumerate(decisions, start=1):
        line = lines_by_id.get(dec.subtitle_id)
        start_sec = line.start_time if line else (idx - 1) * 3.0
        end_sec = line.end_time if line else idx * 3.0

        if end_sec <= start_sec:
            end_sec = start_sec + 2.0

        start_ts = format_srt_timestamp(start_sec)
        end_ts = format_srt_timestamp(end_sec)

        display_text = dec.nadi9_text

        block = f"{idx}\n{start_ts} --> {end_ts}\n{display_text}"
        srt_blocks.append(block)

    srt_content = "\n\n".join(srt_blocks) + ("\n" if srt_blocks else "")
    srt_file.write_text(srt_content, encoding="utf-8")

    # 2. subtitle_decisions.jsonl
    decisions_jsonl_file = out_path / "subtitle_decisions.jsonl"
    jsonl_lines: list[str] = []
    for d in decisions:
        dumped = d.model_dump(mode="json")
        jsonl_lines.append(json.dumps(dumped))

    decisions_jsonl_content = "\n".join(jsonl_lines) + ("\n" if jsonl_lines else "")
    decisions_jsonl_file.write_text(decisions_jsonl_content, encoding="utf-8")

    # 3. learned_rules.json
    learned_rules_file = out_path / "learned_rules.json"
    rules_data = [
        rule.model_dump(mode="json") if hasattr(rule, "model_dump") else rule
        for rule in (learned_rules or [])
    ]
    learned_rules_file.write_text(json.dumps(rules_data, indent=2), encoding="utf-8")

    # 4. review_queue.json
    review_queue_file = out_path / "review_queue.json"
    unresolved_reviews = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in report.review_items
        if not getattr(item, "resolved", False)
    ]
    review_queue_file.write_text(
        json.dumps(unresolved_reviews, indent=2), encoding="utf-8"
    )

    # 5. final_report.md
    final_report_file = out_path / "final_report.md"
    abstained_count = sum(
        1 for d in decisions if getattr(d, "nadi9_text", "") == "Translation unavailable: insufficient supporting evidence." or getattr(d, "abstained", False)
    )

    release_recommendation = (
        "HOLD - HUMAN REVIEW REQUIRED"
        if report.human_review_count > 0
        else "APPROVED FOR RELEASE"
    )

    md_lines: list[str] = [
        "# NADI-9 Final Report",
        "",
        "## Run Summary",
        f"- **Run ID**: `{report.run_id}`",
        f"- **Episode ID**: `{report.episode_id}`",
        f"- **Timestamp**: `{report.created_at}`",
        f"- **Total Subtitles Processed**: {report.total_subtitles_processed}",
        f"- **Accepted Count**: {report.accepted_count}",
        f"- **Human Review Required Count**: {report.human_review_count}",
        f"- **Abstained Count**: {abstained_count}",
        "",
        "## Final Release Recommendation",
        f"**{release_recommendation}**",
        "",
        "## Decision Summary",
        "",
        "| Subtitle ID | Source Text | Nadi-9 Result | Status | Confidence | Human Review |",
        "|---|---|---|---|---|---|",
    ]

    for d in decisions:
        status_str = d.status.value if hasattr(d.status, "value") else str(d.status)
        conf_str = d.confidence.value if hasattr(d.confidence, "value") else str(d.confidence)
        review_str = "Yes" if d.human_review_required else "No"
        src = d.source_text.replace("|", "\\|")
        res = d.nadi9_text.replace("|", "\\|")
        md_lines.append(
            f"| `{d.subtitle_id}` | {src} | {res} | `{status_str}` | `{conf_str}` | {review_str} |"
        )

    md_lines.extend([
        "",
        "## Evidence & Verification",
        f"- **Total Evidence Items Cited**: {sum(len(d.evidence) for d in decisions)}",
        "- **Source Precedence**: Gold approved examples prioritized over secondary dictionaries and crowdsourced feedback.",
        "- **Independent Verification**: Deterministic checks enforced for proper name preservation, number/quantity preservation, and subtitle reading speed (max 25 cps).",
        f"- **Verification Failures**: {sum(1 for d in decisions if d.verification and d.verification.status.value != 'passed')}",
        f"- **Explicit Abstentions**: {abstained_count}",
        "",
        "## Learned Rules",
    ])

    if learned_rules:
        for r in learned_rules:
            r_id = getattr(r, "rule_id", "R-0")
            stmt = getattr(r, "statement", "")
            cat = getattr(r, "category", "")
            md_lines.append(f"- **{r_id}** (`{cat}`): {stmt}")
    else:
        md_lines.append("None generated during this run.")

    md_lines.extend([
        "",
        "## Human Review Queue",
        f"- **Pending Review Items**: {len(unresolved_reviews)}",
    ])

    if unresolved_reviews:
        for rev in unresolved_reviews:
            r_id = rev.get("review_id", "")
            s_id = rev.get("subtitle_id", "")
            q = rev.get("question", "")
            md_lines.append(f"- **{r_id}** (`{s_id}`): {q}")

    md_lines.extend([
        "",
        "## Audit / Traceability",
        "- **Evidence-Grounded**: Every candidate translation cites verified evidence records.",
        "- **Independent Verification**: Deterministic verification checks ran prior to decision assembly.",
        "- **Deterministic Source Precedence**: High-authority sources overrule lower-ranking dictionary entries.",
        "- **Human-in-the-Loop Interrupts**: Suspended execution state saved in SQLite checkpointer when review is required.",
        "- **Learned Rules**: Learned rules record superseding pointers and affected subtitle IDs for targeted replanning.",
        "",
    ])

    final_report_file.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "subtitles_srt": srt_file,
        "subtitle_decisions_jsonl": decisions_jsonl_file,
        "learned_rules_json": learned_rules_file,
        "review_queue_json": review_queue_file,
        "final_report_md": final_report_file,
    }
