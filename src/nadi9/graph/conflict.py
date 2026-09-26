from nadi9.domain.models import Conflict, EvidenceRecord, Hypothesis


class ConflictDetectionError(ValueError):
    """Raised when conflict detection cannot be performed."""


class ConflictDetector:
    """Detects explicit disagreements among evidence items."""

    def detect(
        self,
        hypothesis: Hypothesis,
        evidence: list[EvidenceRecord],
    ) -> list[Conflict]:
        """Return conflicts involving evidence cited by a hypothesis."""

        if not evidence:
            raise ConflictDetectionError(
                "Cannot detect conflicts without evidence."
            )

        evidence_by_id = {
            item.evidence_id: item
            for item in evidence
        }

        supporting_ids = [
            evidence_id
            for evidence_id in hypothesis.supporting_evidence
            if evidence_id in evidence_by_id
        ]

        conflicts: list[Conflict] = []

        for index, first_id in enumerate(supporting_ids):
            first = evidence_by_id[first_id]

            for second_id in supporting_ids[index + 1:]:
                second = evidence_by_id[second_id]

                if not self._are_explicitly_conflicting(
                    first,
                    second,
                ):
                    continue

                conflict_id = self._build_conflict_id(
                    first_id,
                    second_id,
                )

                conflicts.append(
                    Conflict(
                        conflict_id=conflict_id,
                        description=(
                            "Evidence items disagree and require "
                            "further review."
                        ),
                        evidence_ids=[
                            first_id,
                            second_id,
                        ],
                        resolution=None,
                        resolved=False,
                    )
                )

        return conflicts

    def _are_explicitly_conflicting(
        self,
        first: EvidenceRecord,
        second: EvidenceRecord,
    ) -> bool:
        """Detect an explicit conflict marker in evidence metadata."""

        first_conflicts = first.metadata.get(
            "conflicts_with",
            [],
        )

        second_conflicts = second.metadata.get(
            "conflicts_with",
            [],
        )

        if isinstance(first_conflicts, list):
            if second.evidence_id in first_conflicts:
                return True

        if isinstance(second_conflicts, list):
            if first.evidence_id in second_conflicts:
                return True

        return False

    @staticmethod
    def _build_conflict_id(
        first_id: str,
        second_id: str,
    ) -> str:
        """Build a deterministic conflict identifier."""

        ordered_ids = sorted(
            [first_id, second_id],
        )

        return f"CONFLICT-{ordered_ids[0]}-{ordered_ids[1]}"