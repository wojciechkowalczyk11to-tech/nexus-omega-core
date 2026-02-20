"""Google Gemini provider implementation."""

import asyncio
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse

logger = get_logger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini AI provider."""

    # Model mapping for profiles
    PROFILE_MODELS = {
        "eco": "gemini-2.0-flash",
        "smart": "gemini-2.0-flash",
        "deep": "gemini-1.5-pro",
    }

    # Pricing per 1M tokens (USD)
    PRICING = {
        "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash-lite": {"input": 0.0375, "output": 0.15},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    }

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Gemini provider."""
        super().__init__(api_key)
        self._client: genai.Client | None = None
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate completion using Gemini."""
        if not self.is_available() or self._client is None:
            raise ProviderError("Gemini API key not configured", {"provider": "gemini"})

        start_time = time.time()

        try:
            contents, system_instruction = self._convert_messages(messages)

            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                ),
            )

            content = response.text or ""

            try:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0
            except (AttributeError, TypeError):
                input_tokens = sum(len(m["content"].split()) * 2 for m in messages)
                output_tokens = len(content.split()) * 2

            cost_usd = self.calculate_cost(model, input_tokens, output_tokens)
            latency_ms = int((time.time() - start_time) * 1000)

            return ProviderResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                finish_reason="stop",
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
        pricing = self.PRICING.get(model, {"input": 0.075, "output": 0.30})
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

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[genai_types.Content], str | None]:
        """
        Convert OpenAI-style messages to google.genai format.

        Returns:
            Tuple of (contents list, system_instruction string or None)
        """
        contents: list[genai_types.Content] = []
        system_parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append(
                    genai_types.Content(role="user", parts=[genai_types.Part(text=content)])
                )
            elif role == "assistant":
                contents.append(
                    genai_types.Content(role="model", parts=[genai_types.Part(text=content)])
                )

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return contents, system_instruction
