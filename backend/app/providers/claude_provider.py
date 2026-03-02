"""
Anthropic Claude provider implementation.
"""

import time
from typing import Any

from anthropic import AsyncAnthropic

from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse

logger = get_logger(__name__)


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider."""

    # Model mapping for profiles
    PROFILE_MODELS = {
        "eco": "claude-3-5-haiku-20241022",
        "smart": "claude-sonnet-4-20250514",
        "deep": "claude-opus-4-20250918",
    }

    # Pricing per 1M tokens (USD )
    PRICING = {
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-opus-4-20250918": {"input": 15.0, "output": 75.0},
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Claude provider."""
        super().__init__(api_key)
        if self.api_key:
            self.client = AsyncAnthropic(api_key=self.api_key)
        else:
            self.client = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate completion using Claude."""
        if not self.is_available():
            raise ProviderError("Claude API key not configured", {"provider": "claude"})

        start_time = time.time()

        try:
            # Extract system message if present
            system_message = None
            claude_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    claude_messages.append(msg)

            # Create request
            request_params = {
                "model": model,
                "messages": claude_messages,
                "temperature": temperature,
                "max_tokens": 32000 if model == "claude-opus-4-20250918" else max_tokens,
            }

            if system_message:
                request_params["system"] = system_message

            response = await self.client.messages.create(**request_params)

            # Extract response
            content = response.content[0].text
            finish_reason = response.stop_reason

            # Get token counts
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            cost_usd = self.calculate_cost(model, input_tokens, output_tokens)

            latency_ms = int((time.time() - start_time) * 1000)

            return ProviderResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            logger.error("Claude generation error: %s", e, exc_info=True)
            raise ProviderError(
                f"Claude generation failed: {str(e)}",
                {"provider": "claude", "model": model},
            ) from e

    def get_model_for_profile(self, profile: str) -> str:
        """Get Claude model for profile."""
        return self.PROFILE_MODELS.get(profile, self.PROFILE_MODELS["eco"])

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Claude request."""
        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @property
    def name(self) -> str:
        """Provider name."""
        return "claude"

    @property
    def display_name(self) -> str:
        """Display name."""
        return "Anthropic Claude"
