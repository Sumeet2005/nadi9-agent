class Nadi9Error(Exception):
    """Base exception for the Nadi-9 system."""


class ConfigurationError(Nadi9Error):
    """Raised when application configuration is invalid."""


class EvidenceError(Nadi9Error):
    """Raised when evidence cannot be loaded or validated."""


class ProviderError(Nadi9Error):
    """Raised when an external or mock provider fails."""


class BudgetExceededError(Nadi9Error):
    """Raised when a model or tool budget is exhausted."""


class VerificationError(Nadi9Error):
    """Raised when verification cannot be completed safely."""


class ReplanningError(Nadi9Error):
    """Raised when a correction cannot be safely propagated."""