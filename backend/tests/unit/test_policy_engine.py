"""
Unit tests for policy engine.
"""

from datetime import date

import pytest
import pytest_asyncio
from app.core.exceptions import PolicyDeniedError
from app.db.models.user import User
from app.services.policy_engine import PolicyEngine
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_demo_user_has_limited_provider_access(db_session: AsyncSession):
    """Test that DEMO users have limited provider access."""
    policy = PolicyEngine(db_session)

    demo_user = User(
        telegram_id=123456,
        role="DEMO",
        authorized=True,
    )

    # DEMO can access gemini, deepseek, groq
    result = await policy.check_access(demo_user, "chat", provider="gemini")
    assert result.allowed is True

    # DEMO cannot access openai
    with pytest.raises(PolicyDeniedError) as exc_info:
        await policy.check_access(demo_user, "chat", provider="openai")
    assert "FULL_ACCESS" in str(exc_info.value)


@pytest.mark.asyncio
async def test_full_access_user_has_all_providers(db_session: AsyncSession):
    """Test that FULL_ACCESS users can access all providers."""
    policy = PolicyEngine(db_session)

    full_user = User(
        telegram_id=789012,
        role="FULL_ACCESS",
        authorized=True,
    )

    # Can access openai
    result = await policy.check_access(full_user, "chat", provider="openai")
    assert result.allowed is True

    # Can access claude
    result = await policy.check_access(full_user, "chat", provider="claude")
    assert result.allowed is True


@pytest.mark.asyncio
async def test_demo_user_cannot_use_deep_mode(db_session: AsyncSession):
    """Test that DEMO users cannot use DEEP profile."""
    policy = PolicyEngine(db_session)

    demo_user = User(
        telegram_id=345678,
        role="DEMO",
        authorized=True,
    )

    with pytest.raises(PolicyDeniedError) as exc_info:
        await policy.check_access(demo_user, "chat", profile="deep")
    assert "DEEP" in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_chain_for_eco_profile(db_session: AsyncSession):
    """Test provider chain selection for ECO profile."""
    policy = PolicyEngine(db_session)

    chain = policy.get_provider_chain("DEMO", "eco")

    # ECO chain should start with gemini
    assert chain[0] == "gemini"
    assert "groq" in chain
    assert "deepseek" in chain


@pytest.mark.asyncio
async def test_provider_chain_for_deep_profile(db_session: AsyncSession):
    """Test provider chain selection for DEEP profile."""
    policy = PolicyEngine(db_session)

    chain = policy.get_provider_chain("FULL_ACCESS", "deep")

    # DEEP chain should include premium providers
    assert "deepseek" in chain
    assert "openai" in chain or "claude" in chain


@pytest.mark.asyncio
async def test_increment_counter_creates_new_counter(db_session: AsyncSession):
    """Test that increment_counter creates new counter if not exists."""
    policy = PolicyEngine(db_session)

    counter = await policy.increment_counter(
        telegram_id=111222,
        field="smart_credits_used",
        amount=2,
        cost_usd=0.01,
    )

    assert counter.smart_credits_used == 2
    assert counter.total_cost_usd == 0.01
    assert counter.date == date.today()


@pytest.mark.asyncio
async def test_increment_counter_updates_existing_counter(db_session: AsyncSession):
    """Test that increment_counter updates existing counter."""
    policy = PolicyEngine(db_session)

    # Create initial counter
    await policy.increment_counter(
        telegram_id=333444,
        field="grok_calls",
        amount=1,
        cost_usd=0.05,
    )

    # Increment again
    counter = await policy.increment_counter(
        telegram_id=333444,
        field="grok_calls",
        amount=1,
        cost_usd=0.05,
    )

    assert counter.grok_calls == 2
    assert counter.total_cost_usd == 0.10


@pytest.mark.asyncio
async def test_free_provider_detection(db_session: AsyncSession):
    """Test free provider detection."""
    policy = PolicyEngine(db_session)

    assert policy.is_free_provider("groq") is True
    assert policy.is_free_provider("openrouter") is True
    assert policy.is_free_provider("gemini") is False
    assert policy.is_free_provider("openai") is False


# Fixtures
@pytest_asyncio.fixture
async def db_session():
    """Mock database session for testing."""
    from unittest.mock import AsyncMock, MagicMock

    session = AsyncMock(spec=AsyncSession)

    # Track added objects so increment_counter can find them on subsequent calls
    _added_objects: list = []

    def _add_side_effect(obj):
        _added_objects.append(obj)

    def _execute_side_effect(*args, **kwargs):
        mock_result = MagicMock()
        # Return last added ToolCounter if one exists, None otherwise
        found = None
        for obj in _added_objects:
            if hasattr(obj, "grok_calls"):  # ToolCounter
                found = obj
        mock_result.scalar_one_or_none.return_value = found
        return mock_result

    session.execute = AsyncMock(side_effect=_execute_side_effect)
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock(side_effect=_add_side_effect)

    return session
