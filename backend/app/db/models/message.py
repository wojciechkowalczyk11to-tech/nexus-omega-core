"""
Message model for storing conversation messages.
"""

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.agent_trace import AgentTrace
    from app.db.models.session import ChatSession
    from app.db.models.user import User


class Message(Base):
    """Message model for conversation history."""

    __tablename__ = "messages"

    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Message content
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), default="text", nullable=False)

    # Metadata (provider, model, cost, etc.)
    msg_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")
    user: Mapped["User"] = relationship("User", back_populates="messages")
    agent_traces: Mapped[list["AgentTrace"]] = relationship(
        "AgentTrace",
        back_populates="message",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, session_id={self.session_id}, role={self.role})>"
