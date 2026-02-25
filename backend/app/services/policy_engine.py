"""
Policy engine for RBAC, provider access control, and usage limits.
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import BudgetExceededError, PolicyDeniedError
from app.db.models.tool_counter import ToolCounter
from app.db.models.user import User


@dataclass
class PolicyResult:
    """Result of policy check."""

    allowed: bool
    reason: str = ""
    provider_chain: list[str] | None = None


class PolicyEngine:
    """Policy engine for access control and limits."""

    # Provider access matrix: role -> provider -> allowed
    PROVIDER_ACCESS = {
        "DEMO": {
            "gemini": True,  # ECO only
            "deepseek": True,  # Limited to 50/day
            "groq": True,  # Free tier
            "openrouter": True,  # Free tier only
            "grok": True,  # Limited to 5/day
            "openai": False,
            "claude": False,
        },
        "FULL_ACCESS": {
            "gemini": True,
            "deepseek": True,
            "groq": True,
            "openrouter": True,
            "grok": True,
            "openai": True,
            "claude": True,
        },
        "ADMIN": {
            "gemini": True,
            "deepseek": True,
            "groq": True,
            "openrouter": True,
            "grok": True,
            "openai": True,
            "claude": True,
        },
    }

    # Command access matrix: role -> command -> allowed
    COMMAND_ACCESS = {
        "DEMO": {
            "chat": True,
            "mode": True,  # ECO only
            "session": True,
            "memory": True,
            "export": True,
            "usage": True,
            "rag": False,
            "github": False,
            "subscribe": True,
            "admin": False,
        },
        "FULL_ACCESS": {
            "chat": True,
            "mode": True,
            "session": True,
            "memory": True,
            "export": True,
            "usage": True,
            "rag": True,
            "github": True,
            "subscribe": True,
            "admin": False,
        },
        "ADMIN": {
            "chat": True,
            "mode": True,
            "session": True,
            "memory": True,
            "export": True,
            "usage": True,
            "rag": True,
            "github": True,
            "subscribe": True,
            "admin": True,
        },
    }

    # Tool limits for DEMO users (per 24h)
    TOOL_LIMITS_DEMO = {
        "grok_calls": settings.demo_grok_daily,
        "web_calls": settings.demo_web_daily,
        "smart_credits_used": settings.demo_smart_credits_daily,
        "deepseek_calls": settings.demo_deepseek_daily,
    }

    # Budget caps (USD per day)
    BUDGET_CAPS = {
        "DEMO": 0.0,
        "FULL_ACCESS": settings.full_daily_usd_cap,
        "ADMIN": float("inf"),
    }

    # Provider chains by profile
    PROVIDER_CHAINS = {
        "eco": ["gemini", "groq", "deepseek"],
        "smart": ["deepseek", "gemini", "groq"],
        "deep": ["deepseek", "gemini", "openai", "claude"],
    }

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def check_access(
        self,
        user: User,
        action: str,
        provider: str | None = None,
        profile: str = "eco",
    ) -> PolicyResult:
        """
        Check if user has access to perform an action.

        Args:
            user: User instance
            action: Action to check (e.g., "chat", "rag", "github")
            provider: Optional specific provider
            profile: Profile (eco, smart, deep)

        Returns:
            PolicyResult with allowed status and reason

        Raises:
            PolicyDeniedError: If access is denied
        """
        role = user.role

        # Check if user is authorized
        if not user.authorized and role != "ADMIN":
            raise PolicyDeniedError(
                "Musisz odblokować dostęp. Użyj /unlock <kod>",
                {"role": role, "authorized": False},
            )

        # Check subscription expiration
        if role in ("FULL_ACCESS",) and user.subscription_expires_at:
            from datetime import UTC, datetime

            if user.subscription_expires_at < datetime.now(UTC):
                # Subscription expired — degrade to DEMO
                role = "DEMO"

        # Check command access
        if action in self.COMMAND_ACCESS.get(role, {}) and not self.COMMAND_ACCESS[role][action]:
            raise PolicyDeniedError(
                f"Komenda /{action} wymaga roli FULL_ACCESS lub wyższej",
                {"role": role, "action": action},
            )

        # Check profile access
        if profile == "deep" and role == "DEMO":
            raise PolicyDeniedError(
                "Tryb DEEP wymaga roli FULL_ACCESS. Użyj /subscribe",
                {"role": role, "profile": profile},
            )

        # Check provider access if specified
        if provider and not self.PROVIDER_ACCESS.get(role, {}).get(provider, False):
            raise PolicyDeniedError(
                f"Provider {provider} wymaga roli FULL_ACCESS",
                {"role": role, "provider": provider},
            )

        # Check daily limits for DEMO users
        if role == "DEMO":
            await self._check_demo_limits(user.telegram_id)

        # Check budget cap
        await self._check_budget_cap(user.telegram_id, role)

        # Get provider chain for profile
        chain = self.get_provider_chain(role, profile)

        return PolicyResult(
            allowed=True,
            reason="Access granted",
            provider_chain=chain,
        )

    def get_provider_chain(self, role: str, profile: str) -> list[str]:
        """
        Get provider fallback chain for role and profile.

        Args:
            role: User role
            profile: Profile (eco, smart, deep)

        Returns:
            List of provider names in fallback order
        """
        base_chain = self.PROVIDER_CHAINS.get(profile, self.PROVIDER_CHAINS["eco"])

        # Filter chain based on role access
        allowed_providers = [
            p for p in base_chain if self.PROVIDER_ACCESS.get(role, {}).get(p, False)
        ]

        return allowed_providers

    async def _check_demo_limits(self, telegram_id: int) -> None:
        """
        Check if DEMO user has exceeded daily limits.

        Args:
            telegram_id: User's Telegram ID

        Raises:
            PolicyDeniedError: If limits exceeded
        """
        today = date.today()

        # Get today's counter
        result = await self.db.execute(
            select(ToolCounter).where(ToolCounter.user_id == telegram_id, ToolCounter.date == today)
        )
        counter = result.scalar_one_or_none()

        if not counter:
            return  # No usage yet

        # Check each limit
        for field, limit in self.TOOL_LIMITS_DEMO.items():
            current = getattr(counter, field, 0)
            if current >= limit:
                raise PolicyDeniedError(
                    f"Przekroczono dzienny limit: {field} ({current}/{limit}). "
                    f"Odblokuj pełny dostęp: /subscribe",
                    {"field": field, "current": current, "limit": limit},
                )

    async def _check_budget_cap(self, telegram_id: int, role: str) -> None:
        """
        Check if user has exceeded daily budget cap.

        Args:
            telegram_id: User's Telegram ID
            role: User role

        Raises:
            BudgetExceededError: If budget exceeded
        """
        budget_cap = self.BUDGET_CAPS.get(role, 0.0)

        if budget_cap == float("inf"):
            return  # No limit for admins

        today = date.today()

        # Get today's counter
        result = await self.db.execute(
            select(ToolCounter).where(ToolCounter.user_id == telegram_id, ToolCounter.date == today)
        )
        counter = result.scalar_one_or_none()

        if not counter:
            return  # No usage yet

        if counter.total_cost_usd >= budget_cap:
            raise BudgetExceededError(counter.total_cost_usd, budget_cap)

    async def increment_counter(
        self,
        telegram_id: int,
        field: str,
        amount: int = 1,
        cost_usd: float = 0.0,
    ) -> ToolCounter:
        """
        Increment a tool usage counter.

        Args:
            telegram_id: User's Telegram ID
            field: Counter field to increment
            amount: Amount to increment
            cost_usd: Cost to add to total

        Returns:
            Updated ToolCounter instance
        """
        today = date.today()

        # Get or create today's counter
        result = await self.db.execute(
            select(ToolCounter).where(ToolCounter.user_id == telegram_id, ToolCounter.date == today)
        )
        counter = result.scalar_one_or_none()

        if not counter:
            counter = ToolCounter(
                user_id=telegram_id,
                date=today,
                grok_calls=0,
                web_calls=0,
                smart_credits_used=0,
                vertex_queries=0,
                deepseek_calls=0,
                total_cost_usd=0.0,
            )
            self.db.add(counter)
            await self.db.flush()

        # Increment field
        if hasattr(counter, field):
            current_value = getattr(counter, field)
            setattr(counter, field, current_value + amount)

        # Add cost
        counter.total_cost_usd += cost_usd

        await self.db.flush()
        await self.db.refresh(counter)

        return counter

    def is_free_provider(self, provider: str) -> bool:
        """
        Check if provider is free-tier.

        Args:
            provider: Provider name

        Returns:
            True if provider is free-tier
        """
        free_providers = {"groq", "openrouter"}  # Free-tier models only
        return provider in free_providers
