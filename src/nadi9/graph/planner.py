import re
from typing import Sequence

from nadi9.domain.enums import ReviewPriority
from nadi9.domain.models import EpisodeLine, EpisodePlan, EvidenceRecord, SubtitlePriority

# Regular expression signals for risk classification
RELATIONSHIP_KEYWORDS = [
    "elder", "brother", "sister", "father", "mother", "uncle", "aunt",
    "senior", "junior", "boss", "sir", "ji", "saab", "greeting", "farewell",
    "respect", "regards", "elderly"
]

TONE_KEYWORDS = [
    "formal", "informal", "polite", "casual", "slang", "honorific", "register", "tone"
]

CULTURAL_RARE_KEYWORDS = [
    "archaic", "idiom", "proverb", "dialect", "ritual", "custom", "tradition",
    "phrase", "saying", "rare", "unusual"
]


class EpisodePlanner:
    """Deterministic, transparent risk scorer and planner for episode subtitle lines."""

    def plan_episode(
        self,
        episode_id: str,
        episode_lines: Sequence[EpisodeLine],
        evidence: Sequence[EvidenceRecord] | None = None,
    ) -> EpisodePlan:
        """Analyze episode lines and produce a structured execution plan with risk scores."""

        evidence_list = list(evidence) if evidence is not None else None
        priorities: list[SubtitlePriority] = []
        risky_subtitles: list[str] = []
        required_evidence_queries: set[str] = set()
        planned_checks: dict[str, list[str]] = {}

        for line in episode_lines:
            sub_priority = self.evaluate_line_risk(line, evidence_list)
            priorities.append(sub_priority)

            planned_checks[line.subtitle_id] = sub_priority.required_checks

            if sub_priority.priority in (ReviewPriority.HIGH, ReviewPriority.CRITICAL):
                risky_subtitles.append(line.subtitle_id)

            for reason in sub_priority.risk_reasons:
                required_evidence_queries.add(reason)

        # Sort priorities descending by risk score for priority ordering
        priorities.sort(key=lambda p: (-p.risk_score, p.subtitle_id))

        return EpisodePlan(
            episode_id=episode_id,
            priorities=priorities,
            risky_subtitles=risky_subtitles,
            required_evidence=sorted(list(required_evidence_queries)),
            planned_checks=planned_checks,
            status="planned",
        )

    def plan_subset(
        self,
        episode_id: str,
        episode_lines: Sequence[EpisodeLine],
        target_subtitle_ids: Sequence[str],
        evidence: Sequence[EvidenceRecord] | None = None,
    ) -> EpisodePlan:
        """Produce a targeted execution plan strictly for affected/target subtitle IDs."""
        target_set = set(target_subtitle_ids)
        filtered_lines = [line for line in episode_lines if line.subtitle_id in target_set]
        plan = self.plan_episode(
            episode_id=episode_id,
            episode_lines=filtered_lines,
            evidence=evidence,
        )
        plan.status = "targeted_replan"
        return plan

    def evaluate_line_risk(
        self,
        line: EpisodeLine,
        evidence: Sequence[EvidenceRecord] | None = None,
    ) -> SubtitlePriority:
        """Evaluate transparent risk factors for a single subtitle line."""

        text = line.source_text.lower()
        context_str = str(line.context).lower()
        full_text = f"{text} {context_str}"

        risk_score = 0.1  # Base risk
        risk_reasons: list[str] = []
        required_checks: list[str] = ["evidence_support_check", "length_ratio_check"]

        # Factor 1: Relationship & Social Hierarchy Signal (+0.25)
        if any(kw in full_text for kw in RELATIONSHIP_KEYWORDS):
            risk_score += 0.25
            risk_reasons.append("relationship_social_hierarchy")
            required_checks.append("relationship_terminology_check")

        # Factor 2: Tone / Register Sensitivity (+0.20)
        if any(kw in full_text for kw in TONE_KEYWORDS):
            risk_score += 0.20
            risk_reasons.append("tone_register_sensitivity")
            required_checks.append("tone_consistency_check")

        # Factor 3: Proper Names (+0.15)
        words = line.source_text.split()
        capitalized_non_first = [w for w in words[1:] if re.match(r"^[A-Z][a-z]+", w)]
        if capitalized_non_first or "name" in full_text:
            risk_score += 0.15
            risk_reasons.append("proper_names")
            required_checks.append("proper_name_preservation_check")

        # Factor 4: Numbers & Quantities (+0.15)
        if re.search(r"\b\d+\b", line.source_text) or any(num in full_text.split() for num in ["one", "two", "three", "four", "five"]):
            risk_score += 0.15
            risk_reasons.append("numbers_quantities")
            required_checks.append("number_preservation_check")

        # Factor 5: Culturally Specific / Rare Terminology (+0.20)
        if any(kw in full_text for kw in CULTURAL_RARE_KEYWORDS):
            risk_score += 0.20
            risk_reasons.append("cultural_rare_terminology")
            required_checks.append("cultural_context_check")

        # Factor 6: Evidence Availability Signal (+0.15 if evidence list is provided but no matching evidence found)
        if evidence is not None:
            matching_evidence = [
                e for e in evidence
                if any(term in e.content.lower() for term in text.split() if len(term) > 3)
            ]
            if not matching_evidence:
                risk_score += 0.15
                risk_reasons.append("insufficient_existing_evidence")
                required_checks.append("evidence_sufficiency_check")

        # Cap score between 0.0 and 1.0
        risk_score = round(min(1.0, risk_score), 2)

        # Priority Mapping
        if risk_score >= 0.70:
            priority = ReviewPriority.CRITICAL
            requires_deep_reasoning = True
        elif risk_score >= 0.45:
            priority = ReviewPriority.HIGH
            requires_deep_reasoning = True
        elif risk_score >= 0.25:
            priority = ReviewPriority.MEDIUM
            requires_deep_reasoning = False
        else:
            priority = ReviewPriority.LOW
            requires_deep_reasoning = False

        return SubtitlePriority(
            subtitle_id=line.subtitle_id,
            priority=priority,
            risk_score=risk_score,
            risk_reasons=risk_reasons,
            required_checks=sorted(list(set(required_checks))),
            requires_deep_reasoning=requires_deep_reasoning,
        )
