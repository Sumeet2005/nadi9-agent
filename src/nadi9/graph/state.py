from typing import Annotated, TypeVar, TypedDict

from nadi9.domain.models import (
    Conflict,
    EpisodeLine,
    EpisodePlan,
    EvidenceRecord,
    HumanReviewAction,
    Hypothesis,
    LearnedRule,
    ReviewItem,
    SubtitleDecision,
)

T = TypeVar("T")


def append_items(current: list[T], new: list[T]) -> list[T]:
    """Append new state items without replacing existing items."""
    return current + new


def merge_learned_rules(current: list[LearnedRule], new: list[LearnedRule]) -> list[LearnedRule]:
    """Merge learned rules, handling superseding and preserving rule audit history."""
    rule_map: dict[str, LearnedRule] = {r.rule_id: r for r in current}
    for rule in new:
        rule_map[rule.rule_id] = rule
    return list(rule_map.values())


def replace_value(current: T, new: T) -> T:
    """Replace a state value with its latest value."""
    return new


class BudgetState(TypedDict):
    """Tracks model and tool usage for one episode run."""

    model_calls_used: int
    tool_calls_used: int
    max_model_calls: int
    max_tool_calls: int


class AgentState(TypedDict):
    """Central state shared across the Nadi-9 agent workflow."""

    run_id: str
    episode_id: str

    episode_lines: list[EpisodeLine]

    plan: EpisodePlan | None

    evidence: Annotated[list[EvidenceRecord], append_items]
    hypotheses: Annotated[list[Hypothesis], append_items]
    conflicts: Annotated[list[Conflict], append_items]
    review_items: Annotated[list[ReviewItem], append_items]
    human_review_actions: Annotated[list[HumanReviewAction], append_items]
    learned_rules: Annotated[list[LearnedRule], merge_learned_rules]

    subtitle_decisions: Annotated[list[SubtitleDecision], append_items]

    changed_evidence_ids: Annotated[list[str], append_items]
    affected_subtitle_ids: Annotated[list[str], append_items]

    budget: BudgetState

    current_step: str
    errors: Annotated[list[str], append_items]


def create_initial_state(
    run_id: str,
    episode_id: str,
    episode_lines: list[EpisodeLine] | None = None,
    max_model_calls: int = 25,
    max_tool_calls: int = 50,
) -> AgentState:
    """Create a clean initial state for one episode run."""

    if max_model_calls < 1:
        raise ValueError("max_model_calls must be at least 1.")

    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1.")

    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "episode_lines": episode_lines or [],
        "plan": None,
        "evidence": [],
        "hypotheses": [],
        "conflicts": [],
        "review_items": [],
        "human_review_actions": [],
        "learned_rules": [],
        "subtitle_decisions": [],
        "changed_evidence_ids": [],
        "affected_subtitle_ids": [],
        "budget": {
            "model_calls_used": 0,
            "tool_calls_used": 0,
            "max_model_calls": max_model_calls,
            "max_tool_calls": max_tool_calls,
        },
        "current_step": "initialized",
        "errors": [],
    }