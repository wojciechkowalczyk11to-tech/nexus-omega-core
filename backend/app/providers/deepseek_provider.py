"""
DeepSeek provider implementation.
"""

import time
from typing import Any

from openai import AsyncOpenAI

from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse

logger = get_logger(__name__)


class DeepSeekProvider(BaseProvider):
    """DeepSeek AI provider."""

    # Model mapping for profiles
    PROFILE_MODELS = {
        "eco": "deepseek-chat",
        "smart": "deepseek-reasoner",
        "deep": "deepseek-reasoner",
    }

    # Pricing per 1M tokens (USD)
    PRICING = {
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize DeepSeek provider."""
        super().__init__(api_key)
        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
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
        """Generate completion using DeepSeek."""
        if not self.is_available():
            raise ProviderError("DeepSeek API key not configured", {"provider": "deepseek"})

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
            logger.error(f"DeepSeek generation error: {e}", exc_info=True)
            raise ProviderError(
                f"DeepSeek generation failed: {str(e)}",
                {"provider": "deepseek", "model": model},
            ) from e

    def get_model_for_profile(self, profile: str) -> str:
        """Get DeepSeek model for profile."""
        return self.PROFILE_MODELS.get(profile, self.PROFILE_MODELS["eco"])

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for DeepSeek request."""
        pricing = self.PRICING.get(model, {"input": 0.14, "output": 0.28})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @property
    def name(self) -> str:
        """Provider name."""
        return "deepseek"

    @property
    def display_name(self) -> str:
        """Display name."""
        return "DeepSeek"
