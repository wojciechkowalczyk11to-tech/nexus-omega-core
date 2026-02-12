"""
Usage service for logging and tracking AI usage and costs.
"""

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ledger import UsageLedger
from app.db.models.user import User


class UsageService:
    """Service for usage tracking and reporting."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def log_request(
        self,
        user_id: int,
        session_id: int | None,
        provider: str,
        model: str,
        profile: str,
        difficulty: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        tool_costs: dict | None = None,
        latency_ms: int = 0,
        fallback_used: bool = False,
    ) -> UsageLedger:
        """
        Log an AI request to the usage ledger.

        Args:
            user_id: User's Telegram ID
            session_id: Session ID
            provider: Provider name
            model: Model name
            profile: Profile (eco, smart, deep)
            difficulty: Difficulty (easy, medium, hard)
            input_tokens: Input token count
            output_tokens: Output token count
            cost_usd: Cost in USD
            tool_costs: Optional tool cost breakdown
            latency_ms: Latency in milliseconds
            fallback_used: Whether fallback was used

        Returns:
            Created UsageLedger instance
        """
        ledger_entry = UsageLedger(
            user_id=user_id,
            session_id=session_id,
            provider=provider,
            model=model,
            profile=profile,
            difficulty=difficulty,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            tool_costs=tool_costs or {},
            latency_ms=latency_ms,
            fallback_used=fallback_used,
        )

        self.db.add(ledger_entry)
        await self.db.flush()
        await self.db.refresh(ledger_entry)

        return ledger_entry

    async def get_summary(
        self, user_id: int, days: int = 30
    ) -> dict[str, int | float]:
        """
        Get usage summary for a user.

        Args:
            user_id: User's Telegram ID
            days: Number of days to look back

        Returns:
            Summary dict with totals
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        result = await self.db.execute(
            select(
                func.count(UsageLedger.id).label("total_requests"),
                func.sum(UsageLedger.input_tokens).label("total_input_tokens"),
                func.sum(UsageLedger.output_tokens).label("total_output_tokens"),
                func.sum(UsageLedger.cost_usd).label("total_cost_usd"),
                func.avg(UsageLedger.latency_ms).label("avg_latency_ms"),
            ).where(
                UsageLedger.user_id == user_id, UsageLedger.created_at >= cutoff_date
            )
        )

        row = result.one()

        return {
            "total_requests": row.total_requests or 0,
            "total_input_tokens": row.total_input_tokens or 0,
            "total_output_tokens": row.total_output_tokens or 0,
            "total_cost_usd": float(row.total_cost_usd or 0.0),
            "avg_latency_ms": float(row.avg_latency_ms or 0.0),
            "period_days": days,
        }

    async def get_costs_by_provider(
        self, user_id: int, days: int = 30
    ) -> list[dict[str, str | float]]:
        """
        Get cost breakdown by provider.

        Args:
            user_id: User's Telegram ID
            days: Number of days to look back

        Returns:
            List of provider cost breakdowns
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        result = await self.db.execute(
            select(
                UsageLedger.provider,
                func.count(UsageLedger.id).label("requests"),
                func.sum(UsageLedger.cost_usd).label("total_cost"),
            )
            .where(
                UsageLedger.user_id == user_id, UsageLedger.created_at >= cutoff_date
            )
            .group_by(UsageLedger.provider)
            .order_by(func.sum(UsageLedger.cost_usd).desc())
        )

        return [
            {
                "provider": row.provider,
                "requests": row.requests,
                "total_cost_usd": float(row.total_cost or 0.0),
            }
            for row in result.all()
        ]

    async def check_budget(self, user_id: int, budget_cap: float) -> dict[str, float]:
        """
        Check current spending against budget cap.

        Args:
            user_id: User's Telegram ID
            budget_cap: Daily budget cap in USD

        Returns:
            Budget status dict
        """
        today = date.today()
        today_start = datetime.combine(today, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )

        result = await self.db.execute(
            select(func.sum(UsageLedger.cost_usd)).where(
                UsageLedger.user_id == user_id, UsageLedger.created_at >= today_start
            )
        )

        total_today = result.scalar() or 0.0

        return {
            "spent_today_usd": float(total_today),
            "budget_cap_usd": budget_cap,
            "remaining_usd": max(0.0, budget_cap - float(total_today)),
            "percentage_used": (float(total_today) / budget_cap * 100)
            if budget_cap > 0
            else 0.0,
        }

    async def get_leaderboard(self, limit: int = 10) -> list[dict[str, int | float]]:
        """
        Get usage leaderboard (top users by request count).

        Args:
            limit: Number of users to return

        Returns:
            List of user stats
        """
        result = await self.db.execute(
            select(
                UsageLedger.user_id,
                func.count(UsageLedger.id).label("total_requests"),
                func.sum(UsageLedger.cost_usd).label("total_cost"),
            )
            .group_by(UsageLedger.user_id)
            .order_by(func.count(UsageLedger.id).desc())
            .limit(limit)
        )

        return [
            {
                "user_id": row.user_id,
                "total_requests": row.total_requests,
                "total_cost_usd": float(row.total_cost or 0.0),
            }
            for row in result.all()
        ]

    async def get_recent_logs(
        self, user_id: int, limit: int = 20
    ) -> list[UsageLedger]:
        """
        Get recent usage logs for a user.

        Args:
            user_id: User's Telegram ID
            limit: Number of logs to return

        Returns:
            List of UsageLedger instances
        """
        result = await self.db.execute(
            select(UsageLedger)
            .where(UsageLedger.user_id == user_id)
            .order_by(UsageLedger.created_at.desc())
            .limit(limit)
        )

        return list(result.scalars().all())
