from dataclasses import dataclass

from nadi9.domain.errors import BudgetExceededError


@dataclass
class BudgetManager:
    """Enforces model and tool call limits for one episode run."""

    max_model_calls: int = 25
    max_tool_calls: int = 50

    model_calls_used: int = 0
    tool_calls_used: int = 0

    def __post_init__(self) -> None:
        if self.max_model_calls < 1:
            raise ValueError("max_model_calls must be at least 1.")

        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1.")

    @property
    def model_calls_remaining(self) -> int:
        """Return the number of model calls still available."""
        return self.max_model_calls - self.model_calls_used

    @property
    def tool_calls_remaining(self) -> int:
        """Return the number of tool calls still available."""
        return self.max_tool_calls - self.tool_calls_used

    def can_make_model_call(self) -> bool:
        """Return whether another model call is allowed."""
        return self.model_calls_used < self.max_model_calls

    def can_make_tool_call(self) -> bool:
        """Return whether another tool call is allowed."""
        return self.tool_calls_used < self.max_tool_calls

    def consume_model_call(self) -> None:
        """Consume one model-call budget unit."""
        if not self.can_make_model_call():
            raise BudgetExceededError(
                f"Model-call budget exceeded: "
                f"{self.model_calls_used}/{self.max_model_calls}."
            )

        self.model_calls_used += 1

    def consume_tool_call(self) -> None:
        """Consume one tool-call budget unit."""
        if not self.can_make_tool_call():
            raise BudgetExceededError(
                f"Tool-call budget exceeded: "
                f"{self.tool_calls_used}/{self.max_tool_calls}."
            )

        self.tool_calls_used += 1

    def reset(self) -> None:
        """Reset usage counters for a fresh episode run."""
        self.model_calls_used = 0
        self.tool_calls_used = 0

    def snapshot(self) -> dict[str, int]:
        """Return a serializable view of current budget usage."""
        return {
            "model_calls_used": self.model_calls_used,
            "tool_calls_used": self.tool_calls_used,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "model_calls_remaining": self.model_calls_remaining,
            "tool_calls_remaining": self.tool_calls_remaining,
        }