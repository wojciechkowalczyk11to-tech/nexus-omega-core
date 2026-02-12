"""
Groq provider implementation.
"""

import time
from typing import Any

from openai import AsyncOpenAI

from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse

logger = get_logger(__name__)


class GroqProvider(BaseProvider):
    """Groq AI provider (free tier)."""

    # Model mapping for profiles
    PROFILE_MODELS = {
        "eco": "llama-3.3-70b-versatile",
        "smart": "llama-3.3-70b-versatile",
        "deep": "llama-3.3-70b-versatile",
    }

    # Pricing (free tier)
    PRICING = {
        "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
        "llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Groq provider."""
        super().__init__(api_key)
        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
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
        """Generate completion using Groq."""
        if not self.is_available():
            raise ProviderError("Groq API key not configured", {"provider": "groq"})

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
            logger.error(f"Groq generation error: {e}", exc_info=True)
            raise ProviderError(
                f"Groq generation failed: {str(e)}",
                {"provider": "groq", "model": model},
            )

    def get_model_for_profile(self, profile: str) -> str:
        """Get Groq model for profile."""
        return self.PROFILE_MODELS.get(profile, self.PROFILE_MODELS["eco"])

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for Groq request (always free)."""
        return 0.0

    @property
    def name(self) -> str:
        """Provider name."""
        return "groq"

    @property
    def display_name(self) -> str:
        """Display name."""
        return "Groq"
