"""
AuditLog model for tracking admin actions and security events.
"""

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.user import User


class AuditLog(Base):
    """Audit log for tracking administrative actions."""

    __tablename__ = "audit_logs"

    actor_telegram_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Action details
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Request metadata
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)

    # Relationships
    actor: Mapped["User"] = relationship(
        "User",
        back_populates="audit_logs",
        foreign_keys=[actor_telegram_id],
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, actor={self.actor_telegram_id}, action={self.action})>"
