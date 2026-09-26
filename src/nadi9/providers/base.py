from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract interface for all language-model providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_schema: type[Any] | None = None,
    ) -> Any:
        """Generate a response from the configured language model."""
        raise NotImplementedError