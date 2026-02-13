"""
RAG Chunk model with pgvector embeddings for semantic search.
"""

from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.rag_item import RagItem
    from app.db.models.user import User


class RagChunk(Base):
    """
    RAG chunk model for storing document fragments with vector embeddings.

    Each chunk represents a semantic unit of a document with its embedding
    for similarity search using pgvector.
    """

    __tablename__ = "rag_chunks"

    # Foreign keys
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rag_item_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("rag_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Chunk content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Vector embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(384),
        nullable=False,
    )

    # Metadata
    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User")
    rag_item: Mapped["RagItem"] = relationship("RagItem", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<RagChunk(id={self.id}, item_id={self.rag_item_id}, index={self.chunk_index})>"
