"""
Payment model for tracking Telegram Stars payments and subscriptions.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.user import User


class Payment(Base):
    """Payment model for Telegram Stars transactions."""

    __tablename__ = "payments"

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Telegram payment details
    telegram_payment_charge_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    # Product/Plan details
    product_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    plan: Mapped[str] = mapped_column(String(50), nullable=False)  # starter, pro, ultra, enterprise
    tier: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # full_month, full_week, deep_day
    amount_stars: Mapped[int] = mapped_column(Integer, nullable=False)
    stars_amount: Mapped[int] = mapped_column(Integer, nullable=False)  # Alias for compatibility
    credits_granted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="XTR", nullable=False)

    # Provider payment details
    provider_payment_charge_id: Mapped[str] = mapped_column(String(255), nullable=True)

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )  # pending, completed, refunded, failed

    # Subscription period
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="payments")

    def __repr__(self) -> str:
        return f"<Payment(id={self.id}, user_id={self.user_id}, plan={self.plan}, status={self.status})>"
