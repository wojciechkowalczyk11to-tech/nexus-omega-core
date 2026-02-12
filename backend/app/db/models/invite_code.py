"""
InviteCode model for managing user invitations.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.user import User


class InviteCode(Base):
    """Invite code model for user registration."""

    __tablename__ = "invite_codes"

    code_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(50), default="DEMO", nullable=False)

    # Expiration and usage limits
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    uses_left: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Tracking
    created_by: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    consumed_by: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    consumed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    creator: Mapped["User"] = relationship(
        "User",
        back_populates="created_invites",
        foreign_keys=[created_by],
    )
    consumer: Mapped["User | None"] = relationship(
        "User",
        back_populates="consumed_invites",
        foreign_keys=[consumed_by],
    )

    def __repr__(self) -> str:
        return f"<InviteCode(id={self.id}, role={self.role}, uses_left={self.uses_left})>"
