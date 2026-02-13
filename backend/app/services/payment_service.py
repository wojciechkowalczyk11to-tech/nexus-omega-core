"""
Payment service for Telegram Stars integration.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import PaymentError
from app.core.logging_config import get_logger
from app.db.models.payment import Payment
from app.db.models.user import User

logger = get_logger(__name__)


class PaymentService:
    """Service for handling Telegram Stars payments."""

    # Pricing tiers (in Telegram Stars)
    PRICING = {
        "full_access_monthly": {
            "stars": 500,
            "credits": 1000,
            "duration_days": 30,
            "description": "FULL_ACCESS - 30 dni",
        },
        "credits_100": {
            "stars": 50,
            "credits": 100,
            "duration_days": 0,
            "description": "100 kredytów",
        },
        "credits_500": {
            "stars": 200,
            "credits": 500,
            "duration_days": 0,
            "description": "500 kredytów",
        },
        "credits_1000": {
            "stars": 350,
            "credits": 1000,
            "duration_days": 0,
            "description": "1000 kredytów",
        },
    }

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize payment service.

        Args:
            db: Database session
        """
        self.db = db

    async def create_payment(
        self,
        user_id: int,
        product_id: str,
        telegram_payment_charge_id: str,
        provider_payment_charge_id: str,
    ) -> Payment:
        """
        Create payment record.

        Args:
            user_id: User's Telegram ID
            product_id: Product identifier
            telegram_payment_charge_id: Telegram payment charge ID
            provider_payment_charge_id: Provider payment charge ID

        Returns:
            Created Payment instance

        Raises:
            PaymentError: If product not found or payment creation fails
        """
        # Validate product
        if product_id not in self.PRICING:
            raise PaymentError(
                f"Nieznany produkt: {product_id}",
                {"product_id": product_id},
            )

        product = self.PRICING[product_id]

        # Get user
        result = await self.db.execute(
            select(User).where(User.telegram_id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise PaymentError(
                "Użytkownik nie istnieje",
                {"user_id": user_id},
            )

        # Create payment record
        payment = Payment(
            user_id=user_id,
            product_id=product_id,
            plan=product_id,  # Use product_id as plan name
            amount_stars=product["stars"],
            stars_amount=product["stars"],  # Duplicate for compatibility
            credits_granted=product["credits"],
            telegram_payment_charge_id=telegram_payment_charge_id,
            provider_payment_charge_id=provider_payment_charge_id,
            status="completed",
        )

        self.db.add(payment)
        await self.db.flush()

        # Apply benefits
        await self._apply_payment_benefits(user, product)

        await self.db.commit()
        await self.db.refresh(payment)

        logger.info(
            f"Payment created: user={user_id}, product={product_id}, stars={product['stars']}"
        )

        return payment

    async def _apply_payment_benefits(
        self, user: User, product: dict[str, Any]
    ) -> None:
        """
        Apply payment benefits to user.

        Args:
            user: User instance
            product: Product dict
        """
        from datetime import datetime, timedelta, timezone

        # Grant credits
        user.credits_balance += product["credits"]

        # Upgrade role if FULL_ACCESS purchase
        duration_days = product.get("duration_days", 0)
        if duration_days > 0:
            user.role = "FULL_ACCESS"
            user.authorized = True

            # Check if user has active subscription and extend it
            now = datetime.now(timezone.utc)
            if user.subscription_expires_at and user.subscription_expires_at > now:
                # Extend existing subscription
                user.subscription_expires_at += timedelta(days=duration_days)
                logger.info(
                    f"Extended subscription for user {user.telegram_id} by {duration_days} days"
                )
            else:
                # Create new subscription
                user.subscription_expires_at = now + timedelta(days=duration_days)
                logger.info(
                    f"Created new subscription for user {user.telegram_id} for {duration_days} days"
                )

        logger.info(
            f"Applied benefits: user={user.telegram_id}, credits={product['credits']}, role={user.role}"
        )

    async def get_user_payments(self, user_id: int) -> list[Payment]:
        """
        Get user's payment history.

        Args:
            user_id: User's Telegram ID

        Returns:
            List of Payment instances
        """
        result = await self.db.execute(
            select(Payment)
            .where(Payment.user_id == user_id)
            .order_by(Payment.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_payment_by_charge_id(
        self, telegram_payment_charge_id: str
    ) -> Payment | None:
        """
        Get payment by Telegram charge ID.

        Args:
            telegram_payment_charge_id: Telegram payment charge ID

        Returns:
            Payment instance or None
        """
        result = await self.db.execute(
            select(Payment).where(
                Payment.telegram_payment_charge_id == telegram_payment_charge_id
            )
        )
        return result.scalar_one_or_none()

    def get_pricing(self) -> dict[str, dict[str, Any]]:
        """
        Get pricing information.

        Returns:
            Pricing dict
        """
        return self.PRICING
