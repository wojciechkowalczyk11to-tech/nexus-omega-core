"""Test configuration parsing and model names."""

import os
import sys
from pathlib import Path

# Add backend/ to path so we can import app.* modules
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Set required environment variables for Settings initialization
_test_env = {
    "TELEGRAM_BOT_TOKEN": "test_token",
    "DEMO_UNLOCK_CODE": "test_code",
    "BOOTSTRAP_ADMIN_CODE": "test_admin_code",
    "JWT_SECRET_KEY": "test_jwt_secret_key_256bit",
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/test",
    "POSTGRES_PASSWORD": "test_password",
    "REDIS_URL": "redis://localhost:6379/0",
}
for k, v in _test_env.items():
    os.environ.setdefault(k, v)


def test_gemini_provider_models_not_empty() -> None:
    """Verify Gemini provider model strings are not empty."""
    from app.providers.gemini_provider import GeminiProvider

    assert GeminiProvider.PROFILE_MODELS["eco"] != ""
    assert GeminiProvider.PROFILE_MODELS["smart"] != ""
    assert GeminiProvider.PROFILE_MODELS["deep"] != ""


def test_claude_provider_models_not_empty() -> None:
    """Verify Claude provider model strings are not empty."""
    from app.providers.claude_provider import ClaudeProvider

    assert ClaudeProvider.PROFILE_MODELS["eco"] != ""
    assert ClaudeProvider.PROFILE_MODELS["smart"] != ""
    assert ClaudeProvider.PROFILE_MODELS["deep"] != ""


def test_openai_provider_models_not_empty() -> None:
    """Verify OpenAI provider model strings are not empty."""
    from app.providers.openai_provider import OpenAIProvider

    assert OpenAIProvider.PROFILE_MODELS["eco"] != ""
    assert OpenAIProvider.PROFILE_MODELS["smart"] != ""
    assert OpenAIProvider.PROFILE_MODELS["deep"] != ""


def test_grok_provider_models_not_empty() -> None:
    """Verify Grok provider model strings are not empty."""
    from app.providers.grok_provider import GrokProvider

    assert GrokProvider.PROFILE_MODELS["eco"] != ""
    assert GrokProvider.PROFILE_MODELS["smart"] != ""
    assert GrokProvider.PROFILE_MODELS["deep"] != ""


def test_deepseek_provider_models_not_empty() -> None:
    """Verify DeepSeek provider model strings are not empty."""
    from app.providers.deepseek_provider import DeepSeekProvider

    assert DeepSeekProvider.PROFILE_MODELS["eco"] != ""
    assert DeepSeekProvider.PROFILE_MODELS["smart"] != ""
    assert DeepSeekProvider.PROFILE_MODELS["deep"] != ""


def test_slm_router_has_all_tiers() -> None:
    """Verify SLM router has models in all tiers."""
    from app.services.slm_router import ModelTier, SLMRouter

    for tier in ModelTier:
        assert tier in SLMRouter.MODELS, f"Missing tier {tier} in SLM Router"
        assert len(SLMRouter.MODELS[tier]) > 0, f"Empty tier {tier} in SLM Router"


def test_slm_router_models_have_valid_pricing() -> None:
    """Verify all SLM router models have positive pricing."""
    from app.services.slm_router import SLMRouter

    for _tier, models in SLMRouter.MODELS.items():
        for model in models:
            assert model.cost_per_1m_input >= 0, (
                f"Negative input cost for {model.provider}/{model.model}"
            )
            assert model.cost_per_1m_output >= 0, (
                f"Negative output cost for {model.provider}/{model.model}"
            )
            assert model.context_window > 0, (
                f"Invalid context window for {model.provider}/{model.model}"
            )


def test_policy_engine_subscription_tiers() -> None:
    """Verify payment subscription tiers are properly defined."""
    from app.api.v1.routes_payments import SUBSCRIPTION_TIERS

    assert "full_month" in SUBSCRIPTION_TIERS
    assert "full_week" in SUBSCRIPTION_TIERS
    assert "deep_day" in SUBSCRIPTION_TIERS
    for tier_name, config in SUBSCRIPTION_TIERS.items():
        assert "stars" in config, f"Missing stars in tier {tier_name}"
        assert "days" in config, f"Missing days in tier {tier_name}"
        assert "role" in config, f"Missing role in tier {tier_name}"
        assert config["stars"] > 0, f"Invalid stars in tier {tier_name}"
        assert config["days"] > 0, f"Invalid days in tier {tier_name}"
