import re
from dataclasses import dataclass

from nadi9.budget import BudgetManager
from nadi9.domain.models import EvidenceRecord


@dataclass(frozen=True)
class RetrievedEvidence:
    """Evidence item together with its retrieval score."""

    evidence: EvidenceRecord
    score: float


class EvidenceRetriever:
    """Deterministic lexical retriever for evidence records."""

    def __init__(
        self,
        evidence: list[EvidenceRecord],
        budget: BudgetManager | None = None,
    ) -> None:
        self._evidence = list(evidence)
        self._budget = budget

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Convert text into normalized unique tokens."""

        return {
            token
            for token in re.findall(r"\b\w+\b", text.lower())
            if len(token) > 1
        }

    @classmethod
    def _score(
        cls,
        query: str,
        evidence: EvidenceRecord,
    ) -> float:
        """Calculate a normalized lexical-overlap score."""

        query_tokens = cls._tokenize(query)

        if not query_tokens:
            return 0.0

        evidence_tokens = cls._tokenize(
            f"{evidence.evidence_id} "
            f"{evidence.content} "
            f"{evidence.metadata}"
        )

        if not evidence_tokens:
            return 0.0

        overlap = query_tokens & evidence_tokens

        return len(overlap) / len(query_tokens)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        budget: BudgetManager | None = None,
    ) -> list[RetrievedEvidence]:
        """Return the highest-scoring evidence for a query."""

        active_budget = budget or self._budget
        if active_budget is not None:
            active_budget.consume_tool_call()

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0.")

        if not query.strip():
            return []

        scored = [
            RetrievedEvidence(
                evidence=item,
                score=self._score(query, item),
            )
            for item in self._evidence
        ]

        scored = [
            item
            for item in scored
            if item.score >= min_score
        ]

        scored.sort(
            key=lambda item: (
                -item.score,
                item.evidence.evidence_id,
            )
        )

        return scored[:top_k]