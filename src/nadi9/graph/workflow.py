import uuid
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from nadi9.domain.enums import (
    ConfidenceLevel,
    DecisionStatus,
    ReviewPriority,
    VerificationStatus,
)
from nadi9.domain.models import (
    EpisodeLine,
    HumanReviewAction,
    ReviewItem,
    VerificationResult,
)
from nadi9.graph.conflict import ConflictDetector
from nadi9.graph.decision import SubtitleDecisionBuilder
from nadi9.graph.hypothesis import generate_hypothesis
from nadi9.graph.planner import EpisodePlanner
from nadi9.graph.state import AgentState
from nadi9.graph.verification import HypothesisVerifier
from nadi9.providers import LLMProvider
from nadi9.storage.checkpointer import (
    CheckpointError,
    CheckpointNotFoundError,
)


class Nadi9Workflow:
    """LangGraph orchestrator for evidence-grounded subtitle decisions with interrupt/resume HITL support."""

    def __init__(
        self,
        provider: LLMProvider,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.provider = provider
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.planner = EpisodePlanner()
        self.conflict_detector = ConflictDetector()
        self.verifier = HypothesisVerifier()
        self.decision_builder = SubtitleDecisionBuilder()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the stateful LangGraph workflow with checkpointer."""

        builder = StateGraph(AgentState)

        builder.add_node(
            "plan_episode",
            self._node_plan_episode,
        )
        builder.add_node(
            "generate_hypothesis",
            self._node_generate_hypothesis,
        )
        builder.add_node(
            "detect_conflicts",
            self._node_detect_conflicts,
        )
        builder.add_node(
            "verify_decision",
            self._node_verify_decision,
        )
        builder.add_node(
            "generate_decision",
            self._node_generate_decision,
        )
        builder.add_node(
            "route_human_review",
            self._node_route_human_review,
        )
        builder.add_node(
            "finalize",
            self._node_finalize,
        )

        builder.add_edge(START, "plan_episode")
        builder.add_edge("plan_episode", "generate_hypothesis")
        builder.add_edge("generate_hypothesis", "detect_conflicts")
        builder.add_edge("detect_conflicts", "verify_decision")
        builder.add_edge("verify_decision", "generate_decision")

        builder.add_conditional_edges(
            "generate_decision",
            self._route_after_decision,
            {
                "finalize": "finalize",
                "route_human_review": "route_human_review",
            },
        )

        builder.add_edge("route_human_review", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=self.checkpointer)

    def run(
        self,
        initial_state: AgentState,
        thread_id: str | None = None,
    ) -> AgentState:
        """Execute the workflow graph on an initial agent state."""

        sub_id = (
            initial_state["episode_lines"][0].subtitle_id
            if initial_state.get("episode_lines")
            else "sub-001"
        )
        run_id = initial_state.get("run_id", "run-001")
        tid = thread_id or f"{run_id}:{sub_id}"
        config = {"configurable": {"thread_id": tid}}

        return self._graph.invoke(initial_state, config=config)

    def resume(
        self,
        thread_id: str,
        action: dict[str, Any] | HumanReviewAction,
    ) -> AgentState:
        """Resume a statefully suspended workflow execution using a human review action."""

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self._graph.get_state(config)
        if not snapshot or not snapshot.values:
            raise CheckpointNotFoundError(
                f"No active execution checkpoint found for thread '{thread_id}'."
            )

        if not snapshot.next:
            raise CheckpointError(
                f"Thread '{thread_id}' has already completed execution and cannot be resumed."
            )

        action_dict = (
            action.model_dump(mode="json")
            if hasattr(action, "model_dump")
            else action
        )
        return self._graph.invoke(Command(resume=action_dict), config=config)

    def get_state(self, thread_id: str) -> Any:
        """Get the current state snapshot from the checkpointer for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return self._graph.get_state(config)

    def _node_plan_episode(self, state: AgentState) -> dict[str, Any]:
        """Execute structured planning to score risk factors and prioritize checks."""

        episode_id = state.get("episode_id", "ep-001")
        lines = state.get("episode_lines", [])
        evidence = state.get("evidence", [])

        plan = self.planner.plan_episode(
            episode_id=episode_id,
            episode_lines=lines,
            evidence=evidence,
        )

        return {
            "plan": plan,
            "current_step": "planned",
        }

    def _node_generate_hypothesis(self, state: AgentState) -> dict[str, Any]:
        """Generate an evidence-grounded hypothesis using the LLM provider."""

        state_copy = dict(state)
        if state_copy.get("episode") is None and state_copy.get("episode_lines"):
            state_copy["episode"] = state_copy["episode_lines"][0]

        try:
            current_count = len(state.get("hypotheses", []))
            new_state = generate_hypothesis(
                state=state_copy,
                provider=self.provider,
            )
            new_hypotheses = new_state.get("hypotheses", [])[current_count:]
            return {
                "hypotheses": new_hypotheses,
                "current_step": "hypothesis_generated",
            }
        except Exception as exc:
            return {
                "errors": [f"Hypothesis generation failed: {str(exc)}"],
                "current_step": "human_review",
            }

    def _node_detect_conflicts(self, state: AgentState) -> dict[str, Any]:
        """Detect explicit disagreements among supporting evidence."""

        hypotheses = state.get("hypotheses", [])
        evidence = state.get("evidence", [])

        if not hypotheses or not evidence:
            return {"current_step": "conflicts_detected"}

        latest_hypothesis = hypotheses[-1]
        detected = self.conflict_detector.detect(latest_hypothesis, evidence)

        existing_ids = {c.conflict_id for c in state.get("conflicts", [])}
        new_conflicts = [c for c in detected if c.conflict_id not in existing_ids]

        return {
            "conflicts": new_conflicts,
            "current_step": "conflicts_detected",
        }

    def _node_verify_decision(self, state: AgentState) -> dict[str, Any]:
        """Independently verify the latest hypothesis against available evidence."""

        hypotheses = state.get("hypotheses", [])
        evidence = state.get("evidence", [])
        episode = state.get("episode")
        if episode is None and state.get("episode_lines"):
            episode = state["episode_lines"][0]

        if not hypotheses or not evidence:
            verification = VerificationResult(
                status=VerificationStatus.FAILED,
                checks={"supporting_evidence_exists": False},
                failures=["Missing hypothesis or evidence for verification."],
                evidence_checked=[],
                verifier_notes="Verification failed due to missing inputs.",
            )
        else:
            latest_hypothesis = hypotheses[-1]
            verification = self.verifier.verify(latest_hypothesis, evidence, episode_line=episode)

        return {
            "latest_verification": verification,
            "current_step": "decision_verified",
        }

    def _node_generate_decision(self, state: AgentState) -> dict[str, Any]:
        """Build an auditable subtitle decision and create review items if needed."""

        episode = state.get("episode")
        if episode is None and state.get("episode_lines"):
            episode = state["episode_lines"][0]

        if episode is None:
            return {
                "errors": ["Cannot generate decision without an episode line."],
                "current_step": "human_review",
            }

        evidence = state.get("evidence", [])
        hypotheses = state.get("hypotheses", [])
        conflicts = state.get("conflicts", [])

        verification = state.get("latest_verification")
        if verification is None:
            if hypotheses and evidence:
                verification = self.verifier.verify(hypotheses[-1], evidence, episode_line=episode)
            else:
                verification = VerificationResult(
                    status=VerificationStatus.FAILED,
                    checks={"supporting_evidence_exists": False},
                    failures=["No hypothesis or evidence available."],
                    evidence_checked=[],
                    verifier_notes="No verification performed.",
                )

        nadi9_text = state.get("nadi9_text")
        has_no_evidence = not evidence or (verification and verification.status != VerificationStatus.PASSED and "has_supporting_evidence" in verification.checks and not verification.checks["has_supporting_evidence"])

        if not nadi9_text:
            if hypotheses and hypotheses[-1].statement.strip():
                nadi9_text = hypotheses[-1].statement.strip()
            elif has_no_evidence or conflicts or (verification and verification.status != VerificationStatus.PASSED):
                nadi9_text = "Translation unavailable: insufficient supporting evidence."
            else:
                nadi9_text = episode.source_text

        confidence = state.get("confidence")
        if confidence is None:
            if hypotheses:
                confidence = hypotheses[-1].confidence
            else:
                confidence = ConfidenceLevel.LOW

        confidence_reason = state.get("confidence_reason")
        if not confidence_reason:
            if verification.status == VerificationStatus.PASSED and not conflicts:
                confidence_reason = "Evidence-grounded decision verified."
            else:
                confidence_reason = (
                    "Decision requires human review due to verification "
                    "or conflict check."
                )

        try:
            decision = self.decision_builder.build(
                episode=episode,
                nadi9_text=nadi9_text,
                evidence=evidence,
                hypotheses=hypotheses,
                verification=verification,
                conflicts=conflicts,
                confidence=confidence,
                confidence_reason=confidence_reason,
            )

            is_review_needed = decision.human_review_required or bool(state.get("errors"))
            result: dict[str, Any] = {
                "subtitle_decisions": [decision],
                "current_step": "human_review" if is_review_needed else "decision_generated",
            }

            if decision.human_review_required:
                review_item = ReviewItem(
                    review_id=f"REV-{decision.subtitle_id}-{len(state.get('review_items', [])) + 1}",
                    subtitle_id=decision.subtitle_id,
                    reason=decision.confidence_reason or "Human review required.",
                    question=decision.review_question
                    or "Please review this subtitle decision.",
                    evidence_ids=decision.evidence_ids,
                    conflict_ids=decision.conflicts,
                    confidence=decision.confidence,
                    priority=ReviewPriority.HIGH
                    if decision.conflicts
                    else ReviewPriority.MEDIUM,
                    resolved=False,
                )
                result["review_items"] = [review_item]

            return result

        except Exception as exc:
            return {
                "errors": [f"Decision building failed: {str(exc)}"],
                "current_step": "human_review",
            }

    @staticmethod
    def _node_route_human_review(state: AgentState) -> dict[str, Any]:
        """Mark workflow step as human review required and pause via interrupt for human decision."""

        decisions = state.get("subtitle_decisions", [])
        latest_decision = decisions[-1] if decisions else None
        review_items = state.get("review_items", [])
        review_item = review_items[-1] if review_items else None

        interrupt_payload = {
            "run_id": state.get("run_id"),
            "episode_id": state.get("episode_id"),
            "subtitle_id": latest_decision.subtitle_id if latest_decision else None,
            "source_text": latest_decision.source_text if latest_decision else None,
            "nadi9_text": latest_decision.nadi9_text if latest_decision else None,
            "confidence_reason": latest_decision.confidence_reason if latest_decision else None,
            "review_id": review_item.review_id if review_item else None,
        }

        # Invoke LangGraph interrupt to suspend stateful execution
        action_input = interrupt(interrupt_payload)

        # Execution resumes here when Command(resume=action) is passed
        if isinstance(action_input, dict):
            action_type = action_input.get("action_type")
            actor = action_input.get("actor", "human_reviewer")
            reason = action_input.get("reason", "")
            corrected_text = action_input.get("text") or action_input.get("corrected_text")
        elif hasattr(action_input, "action_type"):
            action_type = action_input.action_type
            actor = getattr(action_input, "actor", "human_reviewer")
            reason = getattr(action_input, "reason", "")
            corrected_text = getattr(action_input, "corrected_text", None)
        else:
            raise ValueError(f"Invalid human review action input: {action_input}")

        original_text = latest_decision.nadi9_text if latest_decision else ""

        if latest_decision:
            if action_type == "approve":
                latest_decision.status = DecisionStatus.ACCEPTED
                latest_decision.human_review_required = False
                latest_decision.confidence_reason = (
                    f"Approved by human reviewer ({actor}). "
                    f"Original reason: {latest_decision.confidence_reason}"
                )
            elif action_type == "reject":
                if not reason or not str(reason).strip():
                    raise ValueError("Rejection reason cannot be empty.")
                latest_decision.status = DecisionStatus.REJECTED
                latest_decision.human_review_required = False
                latest_decision.confidence_reason = (
                    f"Rejected by human reviewer ({actor}): {str(reason).strip()}"
                )
            elif action_type == "correct":
                if not corrected_text or not str(corrected_text).strip():
                    raise ValueError("Corrected text cannot be empty.")
                latest_decision.nadi9_text = str(corrected_text).strip()
                latest_decision.status = DecisionStatus.ACCEPTED
                latest_decision.human_review_required = False
                latest_decision.confidence_reason = (
                    f"Corrected by human reviewer ({actor})."
                )
            else:
                raise ValueError(f"Unknown human review action type: {action_type}")

        if review_item:
            review_item.resolved = True

        action_record = HumanReviewAction(
            action_id=f"ACT-{uuid.uuid4().hex[:8]}",
            review_id=review_item.review_id if review_item else f"REV-{latest_decision.subtitle_id if latest_decision else 'sub'}",
            subtitle_id=latest_decision.subtitle_id if latest_decision else "SUB-001",
            action_type=action_type,
            actor=actor,
            reason=str(reason).strip() if reason else ("Human correction supplied." if action_type == "correct" else "Approved AI candidate subtitle."),
            corrected_text=str(corrected_text).strip() if action_type == "correct" and corrected_text else None,
            original_nadi9_text=original_text,
            final_status=latest_decision.status if latest_decision else DecisionStatus.ACCEPTED,
        )

        response: dict[str, Any] = {
            "human_review_actions": [action_record],
            "current_step": "human_review_resolved",
        }

        # If the human reviewer provided a correction, create a deterministic LearnedRule
        if action_type == "correct" and latest_decision:
            from nadi9.domain.models import LearnedRule
            rule = LearnedRule(
                rule_id=f"RULE-CORR-{latest_decision.subtitle_id}-{uuid.uuid4().hex[:6]}",
                category="human_correction",
                statement=f"Prefer '{corrected_text}' over '{original_text}' for subtitle {latest_decision.subtitle_id}",
                supporting_evidence=latest_decision.evidence_ids,
                affected_subtitle_ids=[latest_decision.subtitle_id],
                confidence=ConfidenceLevel.HIGH,
                supersedes=None,
            )
            response["learned_rules"] = [rule]

        return response

    @staticmethod
    def _node_finalize(state: AgentState) -> dict[str, Any]:
        """Mark workflow step as finalized."""
        return {"current_step": "finalize"}

    @staticmethod
    def _route_after_decision(state: AgentState) -> str:
        """Route conditionally based on decision verification and conflict status."""

        if state.get("errors"):
            return "route_human_review"

        decisions = state.get("subtitle_decisions", [])
        if not decisions:
            return "route_human_review"

        latest_decision = decisions[-1]
        if (
            latest_decision.status == DecisionStatus.ACCEPTED
            and not latest_decision.human_review_required
        ):
            return "finalize"

        return "route_human_review"


def build_nadi9_workflow(
    provider: LLMProvider,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Nadi9Workflow:
    """Factory function to build a configured Nadi9Workflow instance."""
    return Nadi9Workflow(provider, checkpointer=checkpointer)
