from typing import Any

from .base import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic provider used for tests and replay mode."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Return the number of generation calls made."""
        return self._call_count

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_schema: type[Any] | None = None,
    ) -> Any:
        """Return the next configured response."""

        self._call_count += 1

        if not self._responses:
            raise RuntimeError(
                "MockLLMProvider has no configured responses."
            )

        response = self._responses.pop(0)

        if response_schema is not None and isinstance(response, dict):
            return response_schema.model_validate(response)

        return response