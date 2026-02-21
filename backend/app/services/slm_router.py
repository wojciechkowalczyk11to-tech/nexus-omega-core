"""
SLM-first Cost-Aware Router — inteligentny routing preferujący małe modele.

Strategia:
1. Rozpocznij od najmniejszego, najtańszego modelu (SLM)
2. Jeśli zadanie jest proste → zostań przy SLM
3. Jeśli zadanie jest złożone → eskaluj do większego modelu
4. Uwzględnij preferencje kosztowe użytkownika

Modele w kolejności eskalacji:
- Tier 0 (Ultra-cheap): Groq Llama 3.1 8B, Gemini Flash
- Tier 1 (Cheap): DeepSeek V3, Gemini Pro
- Tier 2 (Balanced): GPT-4o-mini, Claude Sonnet
- Tier 3 (Premium): GPT-4, Claude Opus
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CostPreference(str, Enum):
    """User's cost preference."""

    LOW = "low"  # Minimize costs, accept lower quality
    BALANCED = "balanced"  # Balance cost and quality
    QUALITY = "quality"  # Prioritize quality over cost


class ModelTier(str, Enum):
    """Model cost/capability tier."""

    ULTRA_CHEAP = "ultra_cheap"  # ~$0.10 per 1M tokens
    CHEAP = "cheap"  # ~$0.50 per 1M tokens
    BALANCED = "balanced"  # ~$2.00 per 1M tokens
    PREMIUM = "premium"  # ~$10.00+ per 1M tokens


@dataclass
class ModelConfig:
    """Model configuration with cost information."""

    provider: str
    model: str
    tier: ModelTier
    cost_per_1m_input: float  # USD
    cost_per_1m_output: float  # USD
    context_window: int
    supports_function_calling: bool
    speed_score: int  # 1-10, higher is faster


class SLMRouter:
    """
    SLM-first cost-aware router.

    Prefers small, fast, cheap models and escalates only when necessary.
    """

    # Model registry ordered by tier
    MODELS = {
        ModelTier.ULTRA_CHEAP: [
            ModelConfig(
                provider="groq",
                model="llama-3.1-8b-instant",
                tier=ModelTier.ULTRA_CHEAP,
                cost_per_1m_input=0.05,
                cost_per_1m_output=0.08,
                context_window=8192,
                supports_function_calling=True,
                speed_score=10,
            ),
            ModelConfig(
                provider="gemini",
                model="gemini-2.0-flash",
                tier=ModelTier.ULTRA_CHEAP,
                cost_per_1m_input=0.10,
                cost_per_1m_output=0.15,
                context_window=32768,
                supports_function_calling=True,
                speed_score=9,
            ),
        ],
        ModelTier.CHEAP: [
            ModelConfig(
                provider="deepseek",
                model="deepseek-chat",
                tier=ModelTier.CHEAP,
                cost_per_1m_input=0.14,
                cost_per_1m_output=0.28,
                context_window=64000,
                supports_function_calling=True,
                speed_score=7,
            ),
            ModelConfig(
                provider="gemini",
                model="gemini-1.5-pro",
                tier=ModelTier.CHEAP,
                cost_per_1m_input=0.50,
                cost_per_1m_output=1.50,
                context_window=128000,
                supports_function_calling=True,
                speed_score=6,
            ),
        ],
        ModelTier.BALANCED: [
            ModelConfig(
                provider="openai",
                model="gpt-4o-mini",
                tier=ModelTier.BALANCED,
                cost_per_1m_input=0.15,
                cost_per_1m_output=0.60,
                context_window=128000,
                supports_function_calling=True,
                speed_score=8,
            ),
            ModelConfig(
                provider="claude",
                model="claude-3-5-sonnet-20241022",
                tier=ModelTier.BALANCED,
                cost_per_1m_input=3.00,
                cost_per_1m_output=15.00,
                context_window=200000,
                supports_function_calling=True,
                speed_score=5,
            ),
        ],
        ModelTier.PREMIUM: [
            ModelConfig(
                provider="openai",
                model="gpt-4-turbo",
                tier=ModelTier.PREMIUM,
                cost_per_1m_input=10.00,
                cost_per_1m_output=30.00,
                context_window=128000,
                supports_function_calling=True,
                speed_score=4,
            ),
            ModelConfig(
                provider="claude",
                model="claude-3-opus-20240229",
                tier=ModelTier.PREMIUM,
                cost_per_1m_input=15.00,
                cost_per_1m_output=75.00,
                context_window=200000,
                supports_function_calling=True,
                speed_score=3,
            ),
        ],
    }

    @classmethod
    def select_model(
        cls,
        difficulty: str,
        cost_preference: CostPreference,
        requires_function_calling: bool = False,
        min_context_window: int = 8192,
    ) -> ModelConfig:
        """
        Select optimal model based on difficulty and cost preference.

        Args:
            difficulty: Task difficulty (simple, moderate, complex)
            cost_preference: User's cost preference
            requires_function_calling: Whether function calling is needed
            min_context_window: Minimum required context window

        Returns:
            Selected model configuration
        """
        # Determine target tier based on difficulty and cost preference
        target_tier = cls._determine_tier(difficulty, cost_preference)

        # Get models from target tier
        candidate_models = cls.MODELS.get(target_tier, [])

        # Filter by requirements
        filtered_models = [
            m
            for m in candidate_models
            if (not requires_function_calling or m.supports_function_calling)
            and m.context_window >= min_context_window
        ]

        if not filtered_models:
            # Fallback to next tier if no models match
            logger.warning(f"No models found in tier {target_tier}, escalating")
            return cls._escalate_tier(target_tier, requires_function_calling, min_context_window)

        # Select fastest model from filtered candidates
        selected = max(filtered_models, key=lambda m: m.speed_score)

        logger.info(
            f"Selected model: {selected.provider}/{selected.model} "
            f"(tier={selected.tier}, difficulty={difficulty}, cost_pref={cost_preference})"
        )

        return selected

    @classmethod
    def _determine_tier(cls, difficulty: str, cost_preference: CostPreference) -> ModelTier:
        """
        Determine target tier based on difficulty and cost preference.

        Strategy:
        - LOW cost preference: Always start with ULTRA_CHEAP, escalate reluctantly
        - BALANCED: Match tier to difficulty
        - QUALITY: Start one tier higher than difficulty suggests
        """
        # Base tier from difficulty
        base_tier_map = {
            "simple": ModelTier.ULTRA_CHEAP,
            "moderate": ModelTier.CHEAP,
            "complex": ModelTier.BALANCED,
        }

        base_tier = base_tier_map.get(difficulty, ModelTier.CHEAP)

        # Adjust based on cost preference
        if cost_preference == CostPreference.LOW:
            # Always prefer cheaper
            tier_order = [
                ModelTier.ULTRA_CHEAP,
                ModelTier.CHEAP,
                ModelTier.BALANCED,
                ModelTier.PREMIUM,
            ]
            base_index = tier_order.index(base_tier)
            return tier_order[max(0, base_index - 1)]

        elif cost_preference == CostPreference.QUALITY:
            # Prefer higher quality
            tier_order = [
                ModelTier.ULTRA_CHEAP,
                ModelTier.CHEAP,
                ModelTier.BALANCED,
                ModelTier.PREMIUM,
            ]
            base_index = tier_order.index(base_tier)
            return tier_order[min(len(tier_order) - 1, base_index + 1)]

        else:  # BALANCED
            return base_tier

    @classmethod
    def _escalate_tier(
        cls,
        current_tier: ModelTier,
        requires_function_calling: bool,
        min_context_window: int,
    ) -> ModelConfig:
        """
        Escalate to next tier when current tier has no suitable models.

        Args:
            current_tier: Current tier
            requires_function_calling: Whether function calling is needed
            min_context_window: Minimum required context window

        Returns:
            Model from next tier
        """
        tier_order = [ModelTier.ULTRA_CHEAP, ModelTier.CHEAP, ModelTier.BALANCED, ModelTier.PREMIUM]
        current_index = tier_order.index(current_tier)

        # Try next tiers
        for i in range(current_index + 1, len(tier_order)):
            next_tier = tier_order[i]
            candidate_models = cls.MODELS.get(next_tier, [])

            filtered_models = [
                m
                for m in candidate_models
                if (not requires_function_calling or m.supports_function_calling)
                and m.context_window >= min_context_window
            ]

            if filtered_models:
                selected = max(filtered_models, key=lambda m: m.speed_score)
                logger.info(f"Escalated to tier {next_tier}: {selected.provider}/{selected.model}")
                return selected

        # Fallback to any model if nothing matches
        logger.error("No suitable model found, using fallback")
        return cls.MODELS[ModelTier.BALANCED][0]

    @classmethod
    def estimate_cost(
        cls,
        model: ModelConfig,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Model configuration
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * model.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * model.cost_per_1m_output
        return input_cost + output_cost

    @classmethod
    def should_escalate(
        cls,
        current_model: ModelConfig,
        task_complexity_score: float,
        quality_threshold: float = 0.7,
    ) -> bool:
        """
        Determine if task should be escalated to higher tier.

        Args:
            current_model: Current model being used
            task_complexity_score: Complexity score (0.0 to 1.0)
            quality_threshold: Minimum quality threshold

        Returns:
            True if should escalate
        """
        # Simple heuristic: if task complexity significantly exceeds model tier
        tier_scores = {
            ModelTier.ULTRA_CHEAP: 0.3,
            ModelTier.CHEAP: 0.5,
            ModelTier.BALANCED: 0.7,
            ModelTier.PREMIUM: 1.0,
        }

        model_capability = tier_scores.get(current_model.tier, 0.5)

        # Escalate if task complexity exceeds model capability by threshold
        should_escalate = task_complexity_score > (model_capability + (1.0 - quality_threshold))

        if should_escalate:
            logger.info(
                f"Recommending escalation: complexity={task_complexity_score:.2f}, "
                f"model_capability={model_capability:.2f}"
            )

        return should_escalate
