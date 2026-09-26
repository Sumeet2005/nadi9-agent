from unittest.mock import MagicMock, patch

import pytest

from nadi9.budget import BudgetManager
from nadi9.domain.errors import ConfigurationError, ProviderError
from nadi9.domain.models import Hypothesis
from nadi9.providers.budgeted import BudgetedLLMProvider
from nadi9.providers.openai import OpenAIProvider


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is not configured"):
        OpenAIProvider()


def test_openai_provider_initializes_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-sk-123")
    with patch("openai.OpenAI") as mock_openai:
        provider = OpenAIProvider()
        assert provider.model == "gpt-4o-mini"
        mock_openai.assert_called_once()


def test_openai_provider_generates_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-sk-123")
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Generated response"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        provider = OpenAIProvider(model="gpt-4o")
        result = provider.generate("Test prompt")

        assert result == "Generated response"
        mock_client.chat.completions.create.assert_called_once()


def test_openai_provider_generates_pydantic_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-sk-123")
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_hypothesis = Hypothesis(
            hypothesis_id="H-001",
            category="greeting",
            statement="Namaskar.",
            supporting_evidence=["E-001"],
        )

        mock_message = MagicMock()
        mock_message.parsed = mock_hypothesis
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_client.beta.chat.completions.parse.return_value.choices = [mock_choice]

        provider = OpenAIProvider()
        result = provider.generate("Test prompt", response_schema=Hypothesis)

        assert isinstance(result, Hypothesis)
        assert result.statement == "Namaskar."
        mock_client.beta.chat.completions.parse.assert_called_once()


def test_openai_provider_handles_api_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-sk-123")
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection timeout")

        provider = OpenAIProvider()
        with pytest.raises(ProviderError, match="OpenAI API call failed"):
            provider.generate("Test prompt")


def test_openai_provider_with_budgeted_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-sk-123")
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Text"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        provider = OpenAIProvider()
        budget_mgr = BudgetManager(max_model_calls=5)
        budgeted = BudgetedLLMProvider(provider, budget_mgr)

        assert budget_mgr.model_calls_used == 0
        budgeted.generate("Test prompt")
        assert budget_mgr.model_calls_used == 1
