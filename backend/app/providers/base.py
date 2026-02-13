"""
Base provider interface for AI providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResponse:
    """Standardized response from AI provider."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    finish_reason: str = "stop"
    raw_response: dict[str, Any] | None = None


class BaseProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize provider.

        Args:
            api_key: API key for the provider
        """
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with role and content
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            ProviderResponse with content and metadata

        Raises:
            ProviderError: If generation fails
        """
        pass

    @abstractmethod
    def get_model_for_profile(self, profile: str) -> str:
        """
        Get model name for a given profile.

        Args:
            profile: Profile name (eco, smart, deep)

        Returns:
            Model identifier string
        """
        pass

    @abstractmethod
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in USD for a request.

        Args:
            model: Model identifier
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in USD
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name."""
        pass

    def is_available(self) -> bool:
        """
        Check if provider is available (has API key).

        Returns:
            True if provider can be used
        """
        return self.api_key is not None and len(self.api_key) > 0
