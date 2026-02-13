"""Google Gemini provider implementation."""

import asyncio
import time
from typing import Any

import google.generativeai as genai

from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse

logger = get_logger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini AI provider."""

    # Model mapping for profiles
    PROFILE_MODELS = {
        "eco": "gemini-2.0-flash-exp",
        "smart": "gemini-2.0-flash-thinking-exp-1219",
        "deep": "gemini-exp-1206",
    }

    # Pricing per 1M tokens (USD)
    PRICING = {
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free tier
        "gemini-2.0-flash-thinking-exp-1219": {"input": 0.0, "output": 0.0},  # Free tier
        "gemini-exp-1206": {"input": 0.0, "output": 0.0},  # Free tier
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Gemini provider."""
        super().__init__(api_key)
        if self.api_key:
            genai.configure(api_key=self.api_key)

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate completion using Gemini."""
        if not self.is_available():
            raise ProviderError("Gemini API key not configured", {"provider": "gemini"})

        start_time = time.time()

        try:
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages(messages)

            # Create model
            model_instance = genai.GenerativeModel(model)

            # Generate
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Run synchronous generate_content in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model_instance.generate_content(
                    gemini_messages,
                    generation_config=generation_config,
                ),
            )

            # Extract response
            content = response.text
            finish_reason = "stop"

            # Get token counts
            try:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            except (AttributeError, KeyError):
                # Fallback estimation
                input_tokens = sum(len(m["content"].split()) * 2 for m in messages)
                output_tokens = len(content.split()) * 2

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
                raw_response={"usage_metadata": response.usage_metadata._pb},
            )

        except Exception as e:
            logger.error(f"Gemini generation error: {e}", exc_info=True)
            raise ProviderError(
                f"Gemini generation failed: {str(e)}",
                {"provider": "gemini", "model": model},
            ) from e

    def get_model_for_profile(self, profile: str) -> str:
        """Get Gemini model for profile."""
        return self.PROFILE_MODELS.get(profile, self.PROFILE_MODELS["eco"])

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Gemini request."""
        pricing = self.PRICING.get(model, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini"

    @property
    def display_name(self) -> str:
        """Display name."""
        return "Google Gemini"

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Convert OpenAI-style messages to Gemini format.

        Args:
            messages: List of message dicts

        Returns:
            Gemini-formatted messages
        """
        gemini_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Map roles
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                gemini_messages.append({"role": "user", "parts": [f"[System] {content}"]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        return gemini_messages
