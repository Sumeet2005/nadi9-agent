import pytest

from nadi9.budget import BudgetManager
from nadi9.domain.errors import BudgetExceededError
from nadi9.providers import BudgetedLLMProvider, MockLLMProvider


def test_budgeted_provider_returns_wrapped_provider_response():
    provider = MockLLMProvider(
        responses=["expected response"]
    )
    budget = BudgetManager(max_model_calls=1)

    budgeted_provider = BudgetedLLMProvider(
        provider=provider,
        budget=budget,
    )

    result = budgeted_provider.generate("test prompt")

    assert result == "expected response"


def test_budgeted_provider_consumes_model_budget():
    provider = MockLLMProvider(
        responses=["response"]
    )
    budget = BudgetManager(max_model_calls=1)

    budgeted_provider = BudgetedLLMProvider(
        provider=provider,
        budget=budget,
    )

    budgeted_provider.generate("test prompt")

    assert budget.model_calls_used == 1
    assert budget.model_calls_remaining == 0


def test_budgeted_provider_blocks_calls_after_budget_is_exhausted():
    provider = MockLLMProvider(
        responses=["first response", "second response"]
    )
    budget = BudgetManager(max_model_calls=1)

    budgeted_provider = BudgetedLLMProvider(
        provider=provider,
        budget=budget,
    )

    assert budgeted_provider.generate("first prompt") == "first response"

    with pytest.raises(
        BudgetExceededError,
        match="Model-call budget exceeded",
    ):
        budgeted_provider.generate("second prompt")

    assert provider.call_count == 1
    assert budget.model_calls_used == 1


def test_budgeted_provider_passes_system_prompt_and_schema():
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        answer: str

    provider = MockLLMProvider(
        responses=[
            {
                "answer": "validated response",
            }
        ]
    )
    budget = BudgetManager(max_model_calls=1)

    budgeted_provider = BudgetedLLMProvider(
        provider=provider,
        budget=budget,
    )

    result = budgeted_provider.generate(
        "test prompt",
        system_prompt="You are a test assistant.",
        response_schema=TestResponse,
    )

    assert isinstance(result, TestResponse)
    assert result.answer == "validated response"


def test_budgeted_provider_exposes_wrapped_provider_and_budget():
    provider = MockLLMProvider(
        responses=["response"]
    )
    budget = BudgetManager(max_model_calls=5)

    budgeted_provider = BudgetedLLMProvider(
        provider=provider,
        budget=budget,
    )

    assert budgeted_provider.provider is provider
    assert budgeted_provider.budget is budget