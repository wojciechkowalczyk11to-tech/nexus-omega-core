"""
Payment endpoints for Telegram Stars subscription management.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin_user, get_current_user, get_db
from app.core.logging_config import get_logger
from app.db.models.payment import Payment
from app.db.models.user import User

logger = get_logger(__name__)

router = APIRouter(prefix="/payments", tags=["payments"])


# Subscription tier definitions
SUBSCRIPTION_TIERS = {
    "full_month": {"stars": 150, "days": 30, "role": "FULL_ACCESS"},
    "full_week": {"stars": 50, "days": 7, "role": "FULL_ACCESS"},
    "deep_day": {"stars": 25, "days": 1, "role": "FULL_ACCESS", "profile_unlock": "deep"},
}


class TelegramStarsVerifyRequest(BaseModel):
    """Request to verify a Telegram Stars payment."""

    telegram_id: int
    tier: str
    stars_paid: int
    telegram_payment_charge_id: str


class TelegramStarsVerifyResponse(BaseModel):
    """Response after verifying a Telegram Stars payment."""

    success: bool
    message: str
    subscription_expires_at: datetime | None = None
    role: str | None = None


class SubscriptionStatusResponse(BaseModel):
    """Subscription status response."""

    telegram_id: int
    role: str
    subscription_tier: str | None
    subscription_expires_at: datetime | None
    is_active: bool


class CancelSubscriptionRequest(BaseModel):
    """Request to cancel a subscription."""

    telegram_id: int
    reason: str = ""


class RevenueAnalytics(BaseModel):
    """Revenue analytics response."""

    total_stars_collected: int
    active_subscriptions: int
    subscriptions_by_tier: dict[str, int]
    total_users: int
    paying_users: int
    arpu: float
    expired_without_renewal: int


@router.post("/telegram-stars/verify", response_model=TelegramStarsVerifyResponse)
async def verify_telegram_stars_payment(
    request: TelegramStarsVerifyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TelegramStarsVerifyResponse:
    """Verify and process a Telegram Stars payment."""
    tier_config = SUBSCRIPTION_TIERS.get(request.tier)
    if not tier_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid subscription tier: {request.tier}",
        )

    if request.stars_paid < tier_config["stars"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient stars: {request.stars_paid} < {tier_config['stars']}",
        )

    # Find user
    result = await db.execute(select(User).where(User.telegram_id == request.telegram_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Calculate expiration
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=tier_config["days"])

    # Extend if already subscribed
    if user.subscription_expires_at and user.subscription_expires_at > now:
        expires_at = user.subscription_expires_at + timedelta(days=tier_config["days"])

    # Update user subscription
    user.role = tier_config["role"]
    user.subscription_tier = request.tier
    user.subscription_expires_at = expires_at

    # Record payment
    payment = Payment(
        user_id=user.telegram_id,
        amount_stars=request.stars_paid,
        stars_amount=request.stars_paid,
        tier=request.tier,
        product_id=f"subscription_{request.tier}",
        plan=request.tier,
        telegram_payment_charge_id=request.telegram_payment_charge_id,
        status="completed",
        expires_at=expires_at,
    )
    db.add(payment)
    await db.flush()

    logger.info(
        f"Payment verified: user={request.telegram_id}, tier={request.tier}, "
        f"stars={request.stars_paid}, expires={expires_at}"
    )

    return TelegramStarsVerifyResponse(
        success=True,
        message=f"Subscription activated: {request.tier} until {expires_at.isoformat()}",
        subscription_expires_at=expires_at,
        role=tier_config["role"],
    )


@router.get("/subscription/status/{telegram_id}", response_model=SubscriptionStatusResponse)
async def get_subscription_status(
    telegram_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SubscriptionStatusResponse:
    """Get subscription status for a user."""
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    now = datetime.now(UTC)
    is_active = user.subscription_expires_at is not None and user.subscription_expires_at > now

    return SubscriptionStatusResponse(
        telegram_id=user.telegram_id,
        role=user.role,
        subscription_tier=user.subscription_tier,
        subscription_expires_at=user.subscription_expires_at,
        is_active=is_active,
    )


@router.post("/subscription/cancel")
async def cancel_subscription(
    request: CancelSubscriptionRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(get_current_admin_user),
) -> dict[str, str]:
    """Cancel a user's subscription (admin only)."""
    result = await db.execute(select(User).where(User.telegram_id == request.telegram_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user.role = "DEMO"
    user.subscription_tier = None
    user.subscription_expires_at = None
    await db.flush()

    logger.info(f"Subscription cancelled: user={request.telegram_id}, reason={request.reason}")

    return {
        "status": "cancelled",
        "message": f"Subscription cancelled for user {request.telegram_id}",
    }


@router.get("/admin/analytics/revenue", response_model=RevenueAnalytics)
async def get_revenue_analytics(
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(get_current_admin_user),
) -> RevenueAnalytics:
    """Get revenue analytics (admin only)."""
    now = datetime.now(UTC)

    # Total stars collected
    total_stars_result = await db.execute(select(func.coalesce(func.sum(Payment.amount_stars), 0)))
    total_stars = total_stars_result.scalar() or 0

    # Active subscriptions
    active_subs_result = await db.execute(
        select(func.count(User.id)).where(
            User.subscription_expires_at > now,
            User.subscription_tier.isnot(None),
        )
    )
    active_subscriptions = active_subs_result.scalar() or 0

    # Subscriptions by tier
    tier_counts: dict[str, int] = {}
    for tier_name in SUBSCRIPTION_TIERS:
        tier_result = await db.execute(
            select(func.count(User.id)).where(
                User.subscription_tier == tier_name,
                User.subscription_expires_at > now,
            )
        )
        tier_counts[tier_name] = tier_result.scalar() or 0

    # Total users
    total_users_result = await db.execute(select(func.count(User.id)))
    total_users = total_users_result.scalar() or 0

    # Paying users (ever paid)
    paying_users_result = await db.execute(select(func.count(func.distinct(Payment.user_id))))
    paying_users = paying_users_result.scalar() or 0

    # ARPU
    arpu = (total_stars / paying_users) if paying_users > 0 else 0.0

    # Expired without renewal (churned)
    expired_result = await db.execute(
        select(func.count(User.id)).where(
            User.subscription_expires_at <= now,
            User.subscription_tier.isnot(None),
        )
    )
    expired_without_renewal = expired_result.scalar() or 0

    return RevenueAnalytics(
        total_stars_collected=total_stars,
        active_subscriptions=active_subscriptions,
        subscriptions_by_tier=tier_counts,
        total_users=total_users,
        paying_users=paying_users,
        arpu=round(arpu, 2),
        expired_without_renewal=expired_without_renewal,
    )
