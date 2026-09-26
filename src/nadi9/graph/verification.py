import re
from typing import Any

from nadi9.domain.enums import VerificationStatus
from nadi9.domain.models import EpisodeLine, EvidenceRecord, Hypothesis, VerificationResult

# Configurable constants for subtitle timing and reading speed verification
MAX_CHARS_PER_SECOND = 25.0
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 12.0

# Common English stop words / titles to ignore during proper name detection
COMMON_NON_NAMES = {
    "The", "A", "An", "In", "On", "At", "To", "For", "With", "By", "About", "Against",
    "Between", "Into", "Through", "During", "Before", "After", "Above", "Below", "From",
    "Up", "Down", "Out", "Off", "Over", "Under", "Again", "Further", "Then", "Once",
    "Here", "There", "When", "Where", "Why", "How", "All", "Any", "Both", "Each",
    "Few", "More", "Most", "Other", "Some", "Such", "No", "Nor", "Not", "Only", "Own",
    "Same", "So", "Than", "Too", "Very", "Can", "Will", "Just", "Don", "Should", "Now"
}


class VerificationError(ValueError):
    """Raised when verification cannot be performed."""


class HypothesisVerifier:
    """Independently verifies an evidence-grounded hypothesis against deterministic linguistic, numeric, and timing rules."""

    def __init__(
        self,
        max_cps: float = MAX_CHARS_PER_SECOND,
        min_duration: float = MIN_DURATION_SECONDS,
        max_duration: float = MAX_DURATION_SECONDS,
    ) -> None:
        self.max_cps = max_cps
        self.min_duration = min_duration
        self.max_duration = max_duration

    def verify(
        self,
        hypothesis: Hypothesis,
        evidence: list[EvidenceRecord],
        episode_line: EpisodeLine | None = None,
    ) -> VerificationResult:
        """Verify that a hypothesis is supported by evidence and satisfies preservation/timing constraints."""

        if not evidence:
            raise VerificationError(
                "Cannot verify a hypothesis without evidence."
            )

        evidence_by_id = {
            item.evidence_id: item
            for item in evidence
        }

        failures: list[str] = []
        checks: dict[str, bool] = {}

        # Check 1: every cited evidence ID exists.
        supporting_ids_valid = all(
            evidence_id in evidence_by_id
            for evidence_id in hypothesis.supporting_evidence
        )
        checks["supporting_evidence_exists"] = supporting_ids_valid
        if not supporting_ids_valid:
            missing_ids = [
                evidence_id
                for evidence_id in hypothesis.supporting_evidence
                if evidence_id not in evidence_by_id
            ]
            failures.append(
                f"Missing supporting evidence: {sorted(missing_ids)}"
            )

        # Check 2: a hypothesis must cite at least one supporting evidence item.
        has_supporting_evidence = bool(hypothesis.supporting_evidence)
        checks["has_supporting_evidence"] = has_supporting_evidence
        if not has_supporting_evidence:
            failures.append("Hypothesis does not cite supporting evidence.")

        # Check 3: hypothesis statement must be non-empty.
        statement_present = bool(hypothesis.statement.strip())
        checks["statement_present"] = statement_present
        if not statement_present:
            failures.append("Hypothesis statement is empty.")

        # Check 4: Proper Name Preservation Check
        if episode_line and hypothesis.statement.strip():
            name_passed, missing_names = self._verify_proper_names(
                source_text=episode_line.source_text,
                proposed_text=hypothesis.statement,
            )
            checks["proper_name_preservation"] = name_passed
            if not name_passed:
                failures.append(
                    f"Proper name preservation check failed. Missing names: {', '.join(missing_names)}"
                )

        # Check 5: Number / Quantity Preservation Check
        if episode_line and hypothesis.statement.strip():
            num_passed, num_diffs = self._verify_numeric_values(
                source_text=episode_line.source_text,
                proposed_text=hypothesis.statement,
            )
            checks["number_preservation"] = num_passed
            if not num_passed:
                failures.append(
                    f"Number preservation check failed. Difference: {num_diffs}"
                )

        # Check 6: Subtitle Timing & Reading Speed Check
        if episode_line and episode_line.end_time > episode_line.start_time:
            timing_passed, timing_reason = self._verify_timing_and_reading_speed(
                line=episode_line,
                proposed_text=hypothesis.statement,
            )
            checks["subtitle_timing"] = timing_passed
            if not timing_passed:
                failures.append(f"Subtitle timing check failed: {timing_reason}")
        else:
            checks["subtitle_timing"] = True

        evidence_checked = [
            evidence_id
            for evidence_id in hypothesis.supporting_evidence
            if evidence_id in evidence_by_id
        ]

        status = VerificationStatus.FAILED if failures else VerificationStatus.PASSED

        return VerificationResult(
            status=status,
            checks=checks,
            failures=failures,
            evidence_checked=evidence_checked,
            verifier_notes="Verification performed using independent deterministic checks.",
        )

    def _verify_proper_names(
        self, source_text: str, proposed_text: str
    ) -> tuple[bool, list[str]]:
        """Extract capitalized entities from source and verify presence in proposed text."""
        words = source_text.split()
        if not words:
            return True, []

        candidates = set()
        for idx, w in enumerate(words):
            cleaned = re.sub(r"[^\w]", "", w)
            if not cleaned or not cleaned[0].isupper():
                continue
            if cleaned in COMMON_NON_NAMES:
                continue
            # If it's the first word, check if it looks like a proper noun (not in common lowercase vocab or length > 2)
            candidates.add(cleaned)

        missing = []
        for name in candidates:
            if not re.search(rf"\b{re.escape(name)}\b", proposed_text, re.IGNORECASE):
                missing.append(name)

        return len(missing) == 0, sorted(missing)

    def _verify_numeric_values(
        self, source_text: str, proposed_text: str
    ) -> tuple[bool, str]:
        """Extract integers, decimals, percentages, currency values and compare equivalence."""
        source_nums = self._extract_normalized_numbers(source_text)
        proposed_nums = self._extract_normalized_numbers(proposed_text)

        if source_nums == proposed_nums:
            return True, ""

        return False, f"Source numbers {source_nums} vs Proposed numbers {proposed_nums}"

    @staticmethod
    def _extract_normalized_numbers(text: str) -> list[float]:
        """Extract and normalize all numeric values from text."""
        raw_matches = re.findall(r"\d+(?:\.\d+)?", text)
        nums = []
        for match in raw_matches:
            try:
                nums.append(float(match))
            except ValueError:
                pass
        return sorted(nums)

    def _verify_timing_and_reading_speed(
        self, line: EpisodeLine, proposed_text: str
    ) -> tuple[bool, str]:
        """Verify subtitle duration and reading speed (characters per second)."""
        duration = line.end_time - line.start_time
        if duration < self.min_duration:
            return False, f"Subtitle duration {duration:.2f}s is below minimum {self.min_duration}s."

        if duration > self.max_duration:
            return False, f"Subtitle duration {duration:.2f}s exceeds maximum {self.max_duration}s."

        char_count = len(proposed_text.strip())
        cps = char_count / duration if duration > 0 else 0
        if cps > self.max_cps:
            return False, f"Reading speed {cps:.1f} cps exceeds maximum allowed {self.max_cps} cps."

        return True, ""