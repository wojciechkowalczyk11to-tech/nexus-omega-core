"""
OpenRouter provider implementation.
"""

import time
from typing import Any

from openai import AsyncOpenAI

from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse

logger = get_logger(__name__)


class OpenRouterProvider(BaseProvider):
    """OpenRouter AI provider (free tier models)."""

    # Model mapping for profiles (free tier only)
    PROFILE_MODELS = {
        "eco": "meta-llama/llama-3.2-3b-instruct:free",
        "smart": "meta-llama/llama-3.1-8b-instruct:free",
        "deep": "meta-llama/llama-3.1-8b-instruct:free",
    }

    # Pricing (free tier)
    PRICING = {
        "meta-llama/llama-3.2-3b-instruct:free": {"input": 0.0, "output": 0.0},
        "meta-llama/llama-3.1-8b-instruct:free": {"input": 0.0, "output": 0.0},
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize OpenRouter provider."""
        super().__init__(api_key)
        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
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
        """Generate completion using OpenRouter."""
        if not self.is_available():
            raise ProviderError("OpenRouter API key not configured", {"provider": "openrouter"})

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

            # Calculate cost (free)
            cost_usd = 0.0

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
            logger.error(f"OpenRouter generation error: {e}", exc_info=True)
            raise ProviderError(
                f"OpenRouter generation failed: {str(e)}",
                {"provider": "openrouter", "model": model},
            ) from e

    def get_model_for_profile(self, profile: str) -> str:
        """Get OpenRouter model for profile."""
        return self.PROFILE_MODELS.get(profile, self.PROFILE_MODELS["eco"])

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenRouter request (always free for free tier)."""
        return 0.0

    @property
    def name(self) -> str:
        """Provider name."""
        return "openrouter"

    @property
    def display_name(self) -> str:
        """Display name."""
        return "OpenRouter"
