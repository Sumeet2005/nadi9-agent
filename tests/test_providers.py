import pytest

from nadi9.providers import LLMProvider, MockLLMProvider


def test_mock_provider_implements_provider_interface():
    provider = MockLLMProvider(responses=["response"])

    assert isinstance(provider, LLMProvider)


def test_mock_provider_returns_configured_response():
    provider = MockLLMProvider(
        responses=["first response", "second response"]
    )

    assert provider.generate("test prompt") == "first response"
    assert provider.generate("test prompt") == "second response"
    assert provider.call_count == 2


def test_mock_provider_is_deterministic():
    provider = MockLLMProvider(
        responses=["expected response"]
    )

    result = provider.generate(
        "Translate this sentence."
    )

    assert result == "expected response"


def test_mock_provider_fails_when_responses_are_exhausted():
    provider = MockLLMProvider(responses=[])

    with pytest.raises(
        RuntimeError,
        match="no configured responses",
    ):
        provider.generate("test prompt")


def test_mock_provider_validates_schema():
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

    result = provider.generate(
        "test prompt",
        response_schema=TestResponse,
    )

    assert isinstance(result, TestResponse)
    assert result.answer == "validated response"


def test_mock_provider_counts_failed_calls():
    provider = MockLLMProvider(responses=[])

    with pytest.raises(RuntimeError):
        provider.generate("test prompt")

    assert provider.call_count == 1