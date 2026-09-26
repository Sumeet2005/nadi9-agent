import os
import time
from typing import Any

from pydantic import BaseModel

from nadi9.domain.errors import ConfigurationError, ProviderError

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider with structured outputs, timeout, and transient retry handling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not configured in environment variables or parameters."
            )

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=resolved_api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to initialize OpenAI client: {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_schema: type[Any] | None = None,
    ) -> Any:
        """Generate text or structured Pydantic object from OpenAI-compatible provider with retry logic."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                if response_schema is not None and isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                    completion = self._client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=response_schema,
                    )
                    message = completion.choices[0].message
                    if message.parsed is not None:
                        return message.parsed
                    if message.content:
                        return response_schema.model_validate_json(message.content)
                    raise ProviderError("OpenAI returned an empty response.")
                else:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                    return response.choices[0].message.content or ""

            except Exception as exc:
                last_exception = exc
                if isinstance(exc, (ConfigurationError, ProviderError)):
                    raise

                if self._is_transient_error(exc) and attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue
                else:
                    break

        raise ProviderError(
            f"OpenAI API call failed: {last_exception}"
        ) from last_exception

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Identify transient network, timeout, or rate-limit errors suitable for retry."""
        exc_str = str(exc).lower()
        transient_keywords = [
            "timeout",
            "connection",
            "rate_limit",
            "ratelimit",
            "500",
            "502",
            "503",
            "504",
            "service unavailable",
        ]
        return any(kw in exc_str for kw in transient_keywords)
