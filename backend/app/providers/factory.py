"""
Provider factory and registry.
"""

from typing import Type

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.core.logging_config import get_logger
from app.providers.base import BaseProvider, ProviderResponse
from app.providers.claude_provider import ClaudeProvider
from app.providers.deepseek_provider import DeepSeekProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.grok_provider import GrokProvider
from app.providers.groq_provider import GroqProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterProvider

logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating AI provider instances."""

    # Provider registry
    PROVIDERS: dict[str, Type[BaseProvider]] = {
        "gemini": GeminiProvider,
        "deepseek": DeepSeekProvider,
        "groq": GroqProvider,
        "openrouter": OpenRouterProvider,
        "grok": GrokProvider,
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
    }

    # Provider name normalization (handle common typos/aliases)
    PROVIDER_ALIASES = {
        "xai": "grok",
        "x.ai": "grok",
        "google": "gemini",
        "anthropic": "claude",
        "llama": "groq",
    }

    @classmethod
    def create(cls, provider_name: str) -> BaseProvider:
        """
        Create provider instance by name.

        Args:
            provider_name: Provider name (e.g., "gemini", "openai")

        Returns:
            Provider instance

        Raises:
            ProviderError: If provider not found or not configured
        """
        # Normalize provider name
        normalized_name = cls.normalize_provider_name(provider_name)

        # Get provider class
        provider_class = cls.PROVIDERS.get(normalized_name)

        if not provider_class:
            raise ProviderError(
                f"Unknown provider: {provider_name}",
                {"provider": provider_name, "normalized": normalized_name},
            )

        # Get API key from settings
        api_key = cls._get_api_key(normalized_name)

        # Create instance
        provider = provider_class(api_key=api_key)

        if not provider.is_available():
            raise ProviderError(
                f"Provider {normalized_name} not configured (missing API key)",
                {"provider": normalized_name},
            )

        return provider

    @classmethod
    def normalize_provider_name(cls, name: str) -> str:
        """
        Normalize provider name (handle aliases and typos).

        Args:
            name: Provider name

        Returns:
            Normalized provider name
        """
        name_lower = name.lower().strip()

        # Check aliases
        if name_lower in cls.PROVIDER_ALIASES:
            return cls.PROVIDER_ALIASES[name_lower]

        return name_lower

    @classmethod
    def _get_api_key(cls, provider_name: str) -> str | None:
        """
        Get API key for provider from settings.

        Args:
            provider_name: Normalized provider name

        Returns:
            API key or None
        """
        key_mapping = {
            "gemini": settings.gemini_api_key,
            "deepseek": settings.deepseek_api_key,
            "groq": settings.groq_api_key,
            "openrouter": settings.openrouter_api_key,
            "grok": settings.xai_api_key,
            "openai": settings.openai_api_key,
            "claude": settings.anthropic_api_key,
        }

        return key_mapping.get(provider_name)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of available (configured) providers.

        Returns:
            List of provider names
        """
        available = []

        for provider_name in cls.PROVIDERS.keys():
            try:
                provider = cls.create(provider_name)
                if provider.is_available():
                    available.append(provider_name)
            except ProviderError:
                continue

        return available

    @classmethod
    async def generate_with_fallback(
        cls,
        provider_chain: list[str],
        messages: list[dict[str, str]],
        profile: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> tuple[ProviderResponse, str, bool]:
        """
        Generate completion with fallback chain.

        Tries providers in order until one succeeds.

        Args:
            provider_chain: List of provider names in fallback order
            messages: Messages to send
            profile: Profile (eco, smart, deep)
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Tuple of (ProviderResponse, provider_name, fallback_used)

        Raises:
            ProviderError: If all providers fail
        """
        last_error = None
        fallback_used = False

        for i, provider_name in enumerate(provider_chain):
            try:
                logger.info(f"Trying provider: {provider_name} (attempt {i+1}/{len(provider_chain)})")

                # Create provider
                provider = cls.create(provider_name)

                # Get model for profile
                model = provider.get_model_for_profile(profile)

                # Generate
                response = await provider.generate(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Mark fallback if not first provider
                if i > 0:
                    fallback_used = True

                logger.info(f"Provider {provider_name} succeeded")
                return response, provider_name, fallback_used

            except ProviderError as e:
                logger.warning(f"Provider {provider_name} failed: {e.message}")
                last_error = e
                continue

        # All providers failed
        from app.core.exceptions import AllProvidersFailedError

        raise AllProvidersFailedError(
            providers_tried=provider_chain,
            last_error=str(last_error) if last_error else "Unknown error",
        )
