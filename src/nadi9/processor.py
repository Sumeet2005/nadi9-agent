from pathlib import Path
from typing import Any
import uuid

from sqlalchemy.orm import Session

from nadi9.audit.report import AuditReport, build_episode_audit_report
from nadi9.domain.enums import DecisionStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, SubtitleDecision, VerificationResult
from nadi9.graph.state import create_initial_state
from nadi9.graph.workflow import Nadi9Workflow
from nadi9.ingestion.jsonl import ingest_episode_lines, ingest_evidence
from nadi9.ingestion.ranking import EvidenceRanker
from nadi9.ingestion.retriever import EvidenceRetriever
from nadi9.providers.base import LLMProvider
from nadi9.providers.budgeted import BudgetedLLMProvider
from nadi9.storage.repositories import (
    AuditRepository,
    DecisionRepository,
    ReviewRepository,
    RunRepository,
)


class EpisodeProcessor:
    """End-to-end processing pipeline for Nadi-9 subtitle decisions with optional database persistence."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_model_calls: int = 25,
        max_tool_calls: int = 50,
        rank_top_k: int = 5,
        db_session: Session | None = None,
        workflow: Nadi9Workflow | None = None,
    ) -> None:
        self.provider = provider
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.rank_top_k = rank_top_k
        self.db_session = db_session
        self.workflow = workflow or Nadi9Workflow(self.provider)

    def process_files(
        self,
        *,
        episode_path: str | Path,
        evidence_path: str | Path,
        run_id: str | None = None,
        episode_id: str | None = None,
    ) -> AuditReport:
        """Process episode lines and evidence loaded directly from JSONL files."""

        episode_lines = ingest_episode_lines(episode_path)
        evidence_records = ingest_evidence(evidence_path)

        return self.process_records(
            episode_lines=episode_lines,
            evidence=evidence_records,
            run_id=run_id,
            episode_id=episode_id,
        )

    def process_records(
        self,
        *,
        episode_lines: list[EpisodeLine],
        evidence: list[EvidenceRecord],
        run_id: str | None = None,
        episode_id: str | None = None,
    ) -> AuditReport:
        """Process in-memory episode lines and evidence records."""

        active_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        active_episode_id = episode_id or (
            episode_lines[0].subtitle_id.split("-")[0]
            if episode_lines
            else "ep-001"
        )

        overall_state = create_initial_state(
            run_id=active_run_id,
            episode_id=active_episode_id,
            max_model_calls=self.max_model_calls,
            max_tool_calls=self.max_tool_calls,
        )

        overall_state["evidence"] = list(evidence)

        # Persistence Repositories (if db_session is attached)
        run_repo = RunRepository(self.db_session) if self.db_session else None
        dec_repo = DecisionRepository(self.db_session) if self.db_session else None
        rev_repo = ReviewRepository(self.db_session) if self.db_session else None
        audit_repo = AuditRepository(self.db_session) if self.db_session else None

        if run_repo:
            run_repo.create_run(active_run_id, active_episode_id)
            run_repo.update_run_status(active_run_id, "RUNNING")

        if audit_repo:
            audit_repo.log_event(
                event_id=f"EV-{uuid.uuid4().hex[:8]}",
                run_id=active_run_id,
                event_type="run_started",
                payload={"episode_id": active_episode_id, "subtitles_count": len(episode_lines)},
            )

        retriever = EvidenceRetriever(evidence)
        ranker = EvidenceRanker()

        processed_dec_by_sub: dict[str, Any] = {}
        if dec_repo:
            existing_decs = dec_repo.list_decisions_for_run(active_run_id)
            processed_dec_by_sub = {d.subtitle_id: d for d in existing_decs}

        for episode_line in episode_lines:
            # Idempotency check: Skip duplicate execution if already processed for this run
            if episode_line.subtitle_id in processed_dec_by_sub:
                existing_model = processed_dec_by_sub[episode_line.subtitle_id]
                existing_dec = SubtitleDecision(
                    subtitle_id=existing_model.subtitle_id,
                    source_text=existing_model.source_text,
                    nadi9_text=existing_model.nadi9_text,
                    status=existing_model.status,
                    confidence=existing_model.confidence,
                    confidence_reason=existing_model.confidence_reason,
                    evidence_ids=existing_model.evidence_ids_json or [],
                    hypothesis_ids=existing_model.hypothesis_ids_json or [],
                    conflicts=existing_model.conflicts_json or [],
                    human_review_required=existing_model.human_review_required,
                    review_question=existing_model.review_question,
                    verification=VerificationResult.model_validate(existing_model.verification_json)
                    if existing_model.verification_json
                    else None,
                    version=existing_model.version,
                    created_at=existing_model.created_at,
                    updated_at=existing_model.updated_at,
                )
                overall_state["subtitle_decisions"].append(existing_dec)
                continue

            # 1. Retrieve and rank relevant evidence for this specific episode line
            retrieved = retriever.search(
                episode_line.source_text,
                top_k=self.rank_top_k,
            )
            ranked_retrieved = ranker.rank(retrieved)
            line_evidence = [item.evidence for item in ranked_retrieved]

            if not line_evidence:
                line_evidence = list(evidence)

            line_state = create_initial_state(
                run_id=active_run_id,
                episode_id=active_episode_id,
                episode_lines=[episode_line],
                max_model_calls=self.max_model_calls,
                max_tool_calls=self.max_tool_calls,
            )
            line_state["evidence"] = line_evidence
            line_state["episode"] = episode_line

            # Execute LangGraph workflow for this line with stable thread_id
            thread_id = f"{active_run_id}:{episode_line.subtitle_id}"
            final_line_state = self.workflow.run(line_state, thread_id=thread_id)

            line_decisions = final_line_state.get("subtitle_decisions", [])
            line_reviews = final_line_state.get("review_items", [])

            # Aggregate outcomes into overall state
            overall_state["hypotheses"].extend(
                final_line_state.get("hypotheses", [])
            )
            overall_state["conflicts"].extend(
                final_line_state.get("conflicts", [])
            )
            overall_state["subtitle_decisions"].extend(line_decisions)
            overall_state["review_items"].extend(line_reviews)
            overall_state["errors"].extend(
                final_line_state.get("errors", [])
            )

            # Persist to database if repositories are present
            if dec_repo and line_decisions:
                for dec in line_decisions:
                    dec_repo.save_decision(active_run_id, dec)

            if rev_repo and line_reviews:
                for rev in line_reviews:
                    rev_repo.save_review_item(active_run_id, rev)

            if audit_repo and line_decisions:
                for dec in line_decisions:
                    audit_repo.log_event(
                        event_id=f"EV-{uuid.uuid4().hex[:8]}",
                        run_id=active_run_id,
                        subtitle_id=dec.subtitle_id,
                        event_type="verification_completed"
                        if dec.status == DecisionStatus.ACCEPTED
                        else "human_review_created",
                        payload={"status": dec.status.value, "nadi9_text": dec.nadi9_text},
                    )

        # Update model call budget usage if provider is BudgetedLLMProvider
        if isinstance(self.provider, BudgetedLLMProvider):
            overall_state["budget"]["model_calls_used"] = (
                self.provider.budget.model_calls_used
            )
            overall_state["budget"]["tool_calls_used"] = (
                self.provider.budget.tool_calls_used
            )

        # Update final run status
        has_review = any(
            r.human_review_required
            or r.status == DecisionStatus.REVIEW_REQUIRED
            for r in overall_state["subtitle_decisions"]
        )
        final_run_status = "WAITING_FOR_REVIEW" if has_review else "COMPLETED"
        if overall_state["errors"] and not overall_state["subtitle_decisions"]:
            final_run_status = "FAILED"

        if run_repo:
            run_repo.update_run_status(
                run_id=active_run_id,
                status=final_run_status,
                budget_info=overall_state["budget"],
                processing_counts={"total": len(episode_lines), "processed": len(overall_state["subtitle_decisions"])},
                error_info={"errors": overall_state["errors"]} if overall_state["errors"] else None,
            )

        if audit_repo:
            audit_repo.log_event(
                event_id=f"EV-{uuid.uuid4().hex[:8]}",
                run_id=active_run_id,
                event_type="run_completed",
                payload={"final_status": final_run_status},
            )

        return build_episode_audit_report(overall_state)

    def resume_human_review(
        self,
        *,
        run_id: str,
        subtitle_id: str,
        action_type: str,
        reason: str | None = None,
        text: str | None = None,
        actor: str = "human_reviewer",
    ) -> SubtitleDecision:
        """Resume a suspended LangGraph workflow for a subtitle decision."""

        thread_id = f"{run_id}:{subtitle_id}"
        action_payload = {
            "action_type": action_type,
            "actor": actor,
            "reason": reason or "",
            "corrected_text": text,
            "text": text,
        }

        resumed_state = self.workflow.resume(thread_id, action_payload)
        decisions = resumed_state.get("subtitle_decisions", [])
        if not decisions:
            raise ValueError(f"No decision found after resuming thread '{thread_id}'.")

        final_decision = decisions[-1]

        if self.db_session:
            dec_repo = DecisionRepository(self.db_session)
            rev_repo = ReviewRepository(self.db_session)
            audit_repo = AuditRepository(self.db_session)
            run_repo = RunRepository(self.db_session)

            dec_repo.save_decision(run_id, final_decision)

            review_items = rev_repo.list_review_items_for_run(run_id)
            for item_model in review_items:
                if item_model.subtitle_id == subtitle_id:
                    item_model.resolved = True

            action_records = resumed_state.get("human_review_actions", [])
            if action_records:
                rev_repo.save_review_action(run_id, action_records[-1])

            past_action = "corrected" if action_type == "correct" else f"{action_type}d"
            audit_repo.log_event(
                event_id=f"EV-{uuid.uuid4().hex[:8]}",
                run_id=run_id,
                subtitle_id=subtitle_id,
                event_type=f"human_review_{past_action}",
                payload={
                    "actor": actor,
                    "action_type": action_type,
                    "final_status": final_decision.status.value,
                },
            )

            remaining_pending = rev_repo.list_review_items_for_run(run_id, pending_only=True)
            if not remaining_pending:
                run_repo.update_run_status(run_id, "COMPLETED")

        return final_decision

    def apply_rule_correction(
        self,
        *,
        overall_state: dict[str, Any],
        new_rule: Any,
        episode_lines: list[EpisodeLine],
        evidence: list[EvidenceRecord],
    ) -> dict[str, Any]:
        """Apply targeted rule correction, selectively invalidating & reprocessing only affected subtitles while preserving unaffected decisions and audit history."""
        affected_ids = set(getattr(new_rule, "affected_subtitle_ids", []) or [])
        if not affected_ids:
            return overall_state

        # Preserve existing learned rules with new rule appended (superseding old rule)
        from nadi9.graph.state import merge_learned_rules
        existing_rules = overall_state.get("learned_rules", [])
        overall_state["learned_rules"] = merge_learned_rules(existing_rules, [new_rule])

        # Selective invalidation: preserve unaffected decisions, filter affected ones for reprocessing
        existing_decisions = list(overall_state.get("subtitle_decisions", []))
        unaffected_decisions: list[SubtitleDecision] = []
        reprocessed_lines: list[EpisodeLine] = []

        for line in episode_lines:
            if line.subtitle_id in affected_ids:
                reprocessed_lines.append(line)
            else:
                for dec in existing_decisions:
                    if dec.subtitle_id == line.subtitle_id:
                        unaffected_decisions.append(dec)
                        break

        # Targeted replanning using EpisodePlanner.plan_subset
        planner = getattr(self.workflow, "planner", None)
        if planner:
            targeted_plan = planner.plan_subset(
                episode_id=overall_state.get("episode_id", "ep-001"),
                episode_lines=episode_lines,
                target_subtitle_ids=list(affected_ids),
                evidence=evidence,
            )
            overall_state["plan"] = targeted_plan

        # Selective re-execution for affected lines only
        retriever = EvidenceRetriever(evidence)
        ranker = EvidenceRanker()
        new_decisions: list[SubtitleDecision] = []

        for line in reprocessed_lines:
            retrieved = retriever.search(line.source_text, top_k=self.rank_top_k)
            line_evidence = [item.evidence for item in ranker.rank(retrieved)] or list(evidence)

            line_state = create_initial_state(
                run_id=overall_state.get("run_id", "run-001"),
                episode_id=overall_state.get("episode_id", "ep-001"),
                episode_lines=[line],
                max_model_calls=self.max_model_calls,
                max_tool_calls=self.max_tool_calls,
            )
            line_state["evidence"] = line_evidence
            line_state["episode"] = line
            line_state["learned_rules"] = overall_state["learned_rules"]

            thread_id = f"{overall_state.get('run_id', 'run-001')}:{line.subtitle_id}:v2"
            final_line_state = self.workflow.run(line_state, thread_id=thread_id)

            line_decs = final_line_state.get("subtitle_decisions", [])
            new_decisions.extend(line_decs)

        # Merge unaffected decisions + reprocessed decisions maintaining line order
        sub_id_to_new = {d.subtitle_id: d for d in new_decisions}
        final_decisions: list[SubtitleDecision] = []

        for line in episode_lines:
            if line.subtitle_id in sub_id_to_new:
                final_decisions.append(sub_id_to_new[line.subtitle_id])
            else:
                for dec in unaffected_decisions:
                    if dec.subtitle_id == line.subtitle_id:
                        final_decisions.append(dec)
                        break

        overall_state["subtitle_decisions"] = final_decisions
        return overall_state
