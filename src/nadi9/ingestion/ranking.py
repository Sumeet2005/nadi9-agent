from dataclasses import dataclass, field
from typing import Any

from nadi9.domain.enums import EvidenceType
from nadi9.domain.models import EvidenceRecord
from nadi9.ingestion.retriever import RetrievedEvidence

# Default source precedence weights mapping evidence sources by authority
DEFAULT_SOURCE_PRECEDENCE: dict[EvidenceType | str, float] = {
    EvidenceType.APPROVED_EXAMPLE: 0.50,  # Highest authority: verified gold examples
    EvidenceType.EXPERT_NOTE: 0.40,       # High authority: expert linguist field notes
    EvidenceType.DICTIONARY_A: 0.30,      # Primary dictionary source
    EvidenceType.DICTIONARY_B: 0.25,      # Secondary dictionary source
    EvidenceType.GRAMMAR: 0.20,           # Grammar rules
    EvidenceType.AUDIO_INTERVIEW: 0.15,   # Audio interviews
    EvidenceType.VIEWER_FEEDBACK: 0.05,  # Lowest authority: raw crowdsourced feedback
    EvidenceType.EPISODE: 0.10,
}

DEFAULT_RELIABILITY_BONUS: dict[str, float] = {
    "high": 0.20,
    "medium": 0.10,
    "low": 0.0,
    "unverified": -0.10,
}


@dataclass(frozen=True)
class EvidenceRankingPolicy:
    """Configurable policy for deterministic source-precedence ranking."""

    type_weights: dict[EvidenceType | str, float] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_PRECEDENCE)
    )
    provenance_bonus: float = 0.05
    reliability_bonus: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RELIABILITY_BONUS)
    )

    def weight_for_type(self, evidence_type: EvidenceType | str) -> float:
        """Return the configured weight for an evidence type."""
        str_val = str(evidence_type.value if isinstance(evidence_type, EvidenceType) else evidence_type)
        for key, weight in self.type_weights.items():
            key_str = str(key.value if isinstance(key, EvidenceType) else key)
            if key_str == str_val:
                return weight
        return 0.0

    def score_breakdown(
        self,
        item: RetrievedEvidence,
    ) -> dict[str, Any]:
        """Calculate detailed auditable score breakdown for an evidence item."""
        evidence = item.evidence
        retrieval_score = item.score

        type_weight = self.weight_for_type(evidence.evidence_type)
        provenance_weight = self.provenance_bonus if (evidence.provenance and evidence.provenance.original_file) else 0.0

        reliability = evidence.metadata.get("reliability")
        reliability_weight = 0.0
        if isinstance(reliability, str):
            reliability_weight = self.reliability_bonus.get(reliability.lower(), 0.0)

        # Recency bonus: newer year in metadata adds up to 0.05 bonus
        recency_weight = 0.0
        date_val = evidence.metadata.get("date") or evidence.metadata.get("year")
        if date_val and str(date_val).isdigit():
            year = int(str(date_val))
            if year >= 2020:
                recency_weight = 0.05
            elif year >= 2010:
                recency_weight = 0.02

        total_score = round(
            retrieval_score + type_weight + provenance_weight + reliability_weight + recency_weight,
            4,
        )

        return {
            "evidence_id": evidence.evidence_id,
            "evidence_type": str(evidence.evidence_type),
            "retrieval_score": round(retrieval_score, 4),
            "type_precedence_weight": round(type_weight, 4),
            "provenance_weight": round(provenance_weight, 4),
            "reliability_weight": round(reliability_weight, 4),
            "recency_weight": round(recency_weight, 4),
            "total_score": total_score,
            "precedence_rationale": (
                f"Source precedence type weight (+{type_weight:.2f}) & reliability (+{reliability_weight:.2f}) "
                f"applied over base query match ({retrieval_score:.2f})."
            ),
        }

    def score(
        self,
        item: RetrievedEvidence,
    ) -> float:
        """Calculate the final evidence precedence score."""
        return self.score_breakdown(item)["total_score"]


class EvidenceRanker:
    """Ranks retrieved evidence using a deterministic source-precedence policy."""

    def __init__(
        self,
        policy: EvidenceRankingPolicy | None = None,
    ) -> None:
        self._policy = policy or EvidenceRankingPolicy()

    @property
    def policy(self) -> EvidenceRankingPolicy:
        """Return the active ranking policy."""
        return self._policy

    def rank(
        self,
        results: list[RetrievedEvidence],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedEvidence]:
        """Rank retrieved evidence deterministically by source precedence and relevance."""

        ranked = sorted(
            results,
            key=lambda item: (
                -self._policy.score(item),
                -item.score,
                item.evidence.evidence_id,
            ),
        )

        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be at least 1.")

            return ranked[:top_k]

        return ranked

    def explain_ranking(
        self,
        results: list[RetrievedEvidence],
    ) -> list[dict[str, Any]]:
        """Return auditable rank order explanations showing why each source was selected or preferred."""

        ranked = self.rank(results)
        explanations: list[dict[str, Any]] = []

        for rank_idx, item in enumerate(ranked, start=1):
            breakdown = self._policy.score_breakdown(item)
            breakdown["rank"] = rank_idx
            explanations.append(breakdown)

        return explanations