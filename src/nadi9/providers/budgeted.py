from typing import Any

from nadi9.budget import BudgetManager

from .base import LLMProvider


class BudgetedLLMProvider(LLMProvider):
    """LLM provider decorator that enforces a model-call budget."""

    def __init__(
        self,
        provider: LLMProvider,
        budget: BudgetManager,
    ) -> None:
        self._provider = provider
        self._budget = budget

    @property
    def budget(self) -> BudgetManager:
        """Return the budget manager used by this provider."""
        return self._budget

    @property
    def provider(self) -> LLMProvider:
        """Return the wrapped provider."""
        return self._provider

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_schema: type[Any] | None = None,
    ) -> Any:
        """Consume one model-call budget unit and invoke the provider."""
        self._budget.consume_model_call()

        return self._provider.generate(
            prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
        )