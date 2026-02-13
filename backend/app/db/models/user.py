"""
User model representing Telegram users with roles and subscriptions.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.audit_log import AuditLog
    from app.db.models.invite_code import InviteCode
    from app.db.models.ledger import UsageLedger
    from app.db.models.message import Message
    from app.db.models.payment import Payment
    from app.db.models.rag_item import RagItem
    from app.db.models.session import ChatSession
    from app.db.models.tool_counter import ToolCounter
    from app.db.models.user_memory import UserMemory


class User(Base):
    """User model with RBAC and subscription management."""

    __tablename__ = "users"

    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # RBAC: DEMO, FULL_ACCESS, ADMIN
    role: Mapped[str] = mapped_column(String(50), default="DEMO", nullable=False, index=True)

    # Access control
    authorized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Subscription
    subscription_tier: Mapped[str | None] = mapped_column(String(50), nullable=True)
    subscription_expires_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Preferences
    default_mode: Mapped[str] = mapped_column(String(50), default="eco", nullable=False)
    cost_preference: Mapped[str] = mapped_column(
        String(50), default="balanced", nullable=False
    )  # low, balanced, quality
    settings: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Relationships
    sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    usage_ledger: Mapped[list["UsageLedger"]] = relationship(
        "UsageLedger",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    tool_counters: Mapped[list["ToolCounter"]] = relationship(
        "ToolCounter",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="actor",
        foreign_keys="AuditLog.actor_telegram_id",
        cascade="all, delete-orphan",
    )
    created_invites: Mapped[list["InviteCode"]] = relationship(
        "InviteCode",
        back_populates="creator",
        foreign_keys="InviteCode.created_by",
        cascade="all, delete-orphan",
    )
    consumed_invites: Mapped[list["InviteCode"]] = relationship(
        "InviteCode",
        back_populates="consumer",
        foreign_keys="InviteCode.consumed_by",
    )
    rag_items: Mapped[list["RagItem"]] = relationship(
        "RagItem",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    memories: Mapped[list["UserMemory"]] = relationship(
        "UserMemory",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    payments: Mapped[list["Payment"]] = relationship(
        "Payment",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User(telegram_id={self.telegram_id}, role={self.role})>"
