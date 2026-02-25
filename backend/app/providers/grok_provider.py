"""
xAI Grok provider implementation.
"""

import time
from typing import Any

from openai import AsyncOpenAI

from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse

logger = get_logger(__name__)


class GrokProvider(BaseProvider):
    """xAI Grok provider."""

    # Model mapping for profiles
    PROFILE_MODELS = {
        "eco": "grok-3-mini-fast",
        "smart": "grok-3-fast",
        "deep": "grok-3",
    }

    # Pricing per 1M tokens (USD)
    PRICING = {
        "grok-3-mini-fast": {"input": 0.30, "output": 0.50},
        "grok-3-fast": {"input": 5.0, "output": 15.0},
        "grok-3": {"input": 3.0, "output": 15.0},
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Grok provider."""
        super().__init__(api_key)
        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1",
            )
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
        """Generate completion using Grok."""
        if not self.is_available():
            raise ProviderError("Grok API key not configured", {"provider": "grok"})

        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract response
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Get token counts
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

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
            logger.error(f"Grok generation error: {e}", exc_info=True)
            raise ProviderError(
                f"Grok generation failed: {str(e)}",
                {"provider": "grok", "model": model},
            ) from e

    def get_model_for_profile(self, profile: str) -> str:
        """Get Grok model for profile."""
        return self.PROFILE_MODELS.get(profile, self.PROFILE_MODELS["eco"])

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Grok request."""
        pricing = self.PRICING.get(model, {"input": 5.0, "output": 15.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @property
    def name(self) -> str:
        """Provider name."""
        return "grok"

    @property
    def display_name(self) -> str:
        """Display name."""
        return "xAI Grok"
