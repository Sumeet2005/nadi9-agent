from enum import StrEnum


class EvidenceType(StrEnum):

    APPROVED_EXAMPLE = "approved_example"

    AUDIO_INTERVIEW = "audio_interview"

    GRAMMAR = "grammar"

    DICTIONARY_A = "dictionary_a"

    DICTIONARY_B = "dictionary_b"

    EXPERT_NOTE = "expert_note"

    VIEWER_FEEDBACK = "viewer_feedback"

    EPISODE = "episode"


class EvidenceStatus(StrEnum):

    ACTIVE = "active"

    CONTESTED = "contested"

    SUPERSEDED = "superseded"

    INVALIDATED = "invalidated"


class HypothesisStatus(StrEnum):

    PROPOSED = "proposed"

    SUPPORTED = "supported"

    CONTESTED = "contested"

    REJECTED = "rejected"

    UNKNOWN = "unknown"


class DecisionStatus(StrEnum):

    DRAFT = "draft"

    TRANSLATED = "translated"

    VERIFYING = "verifying"

    ACCEPTED = "accepted"

    REJECTED = "rejected"

    HUMAN_REVIEW = "human_review"

    REVIEW_REQUIRED = "human_review"


class ConfidenceLevel(StrEnum):

    HIGH = "high"

    MEDIUM = "medium"

    LOW = "low"

    UNKNOWN = "unknown"


class VerificationStatus(StrEnum):

    PASSED = "passed"

    FAILED = "failed"

    UNCERTAIN = "uncertain"


class ReviewPriority(StrEnum):

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"


class AuditEventType(StrEnum):

    RUN_STARTED = "run_started"

    EVIDENCE_INGESTED = "evidence_ingested"

    HYPOTHESIS_CREATED = "hypothesis_created"

    HYPOTHESIS_UPDATED = "hypothesis_updated"

    TRANSLATION_CREATED = "translation_created"

    VERIFICATION_STARTED = "verification_started"

    VERIFICATION_COMPLETED = "verification_completed"

    HUMAN_REVIEW_CREATED = "human_review_created"

    HUMAN_REVIEW_APPROVED = "human_review_approved"

    HUMAN_REVIEW_REJECTED = "human_review_rejected"

    CORRECTION_RECEIVED = "correction_received"

    REPLAN_STARTED = "replan_started"

    RUN_COMPLETED = "run_completed"

    ERROR = "error"