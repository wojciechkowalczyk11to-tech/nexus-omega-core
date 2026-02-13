"""
RagItem model for tracking uploaded documents and their indexing status.
"""

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.rag_chunk import RagChunk
    from app.db.models.user import User


class RagItem(Base):
    """RAG item model for document tracking."""

    __tablename__ = "rag_items"

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Scope and source
    scope: Mapped[str] = mapped_column(String(50), default="user", nullable=False)  # user, global
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # file, url, text
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # File details
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # Indexing status
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )  # pending, processing, indexed, failed

    # Metadata
    item_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="rag_items")
    chunks: Mapped[list["RagChunk"]] = relationship(
        "RagChunk",
        back_populates="rag_item",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<RagItem(id={self.id}, filename={self.filename}, status={self.status})>"
