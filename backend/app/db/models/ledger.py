"""
UsageLedger model for tracking AI provider usage and costs.
"""

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.user import User


class UsageLedger(Base):
    """Usage ledger for tracking AI requests and costs."""

    __tablename__ = "usage_ledger"

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    # Provider details
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    profile: Mapped[str] = mapped_column(String(50), nullable=False)  # eco, smart, deep
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False)  # easy, medium, hard

    # Token usage
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Cost tracking
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    tool_costs: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Performance metrics
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="usage_ledger")

    def __repr__(self) -> str:
        return f"<UsageLedger(id={self.id}, user_id={self.user_id}, provider={self.provider}, cost=${self.cost_usd:.4f})>"
