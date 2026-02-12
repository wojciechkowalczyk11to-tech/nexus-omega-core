"""
ToolCounter model for tracking daily tool usage limits.
"""

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Date, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.user import User


class ToolCounter(Base):
    """Daily tool usage counter for enforcing limits."""

    __tablename__ = "tool_counters"
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_tool_counter_user_date"),)

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Tool usage counters
    grok_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    web_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    smart_credits_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    vertex_queries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    deepseek_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Total cost tracking
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="tool_counters")

    def __repr__(self) -> str:
        return f"<ToolCounter(user_id={self.user_id}, date={self.date}, smart_credits={self.smart_credits_used})>"
