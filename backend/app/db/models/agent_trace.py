"""
Agent Trace model for storing agent's reasoning steps.
"""

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.message import Message
    from app.db.models.user import User


class AgentTrace(Base):
    """
    Agent trace model for storing reasoning steps.

    Stores the complete thought process of the agent for a given message,
    including reasoning, actions, observations, and tool calls.
    """

    __tablename__ = "agent_traces"

    # Foreign keys
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Trace data
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # think, use_tool, respond, self_correct
    thought: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tool_args: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tool_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    correction_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User")
    message: Mapped["Message"] = relationship("Message", back_populates="agent_traces")

    def __repr__(self) -> str:
        return f"<AgentTrace(id={self.id}, message_id={self.message_id}, iteration={self.iteration}, action={self.action})>"
