"""
RAG tool for document upload, chunking, and semantic search.
"""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import RAGError
from app.core.logging_config import get_logger
from app.db.models.rag_item import RagItem

logger = get_logger(__name__)


class RAGTool:
    """Tool for RAG document management."""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".json"}

    # Chunk size for text splitting
    CHUNK_SIZE = 1000  # characters
    CHUNK_OVERLAP = 200  # characters

    def __init__(self, db: AsyncSession, storage_path: str | None = None) -> None:
        """
        Initialize RAG tool.

        Args:
            db: Database session
            storage_path: Path to store uploaded files
        """
        self.db = db
        self.storage_path = storage_path or "/tmp/rag_storage"

        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    async def upload_document(
        self,
        user_id: int,
        filename: str,
        content: bytes,
        scope: str = "user",
        source_url: str | None = None,
    ) -> RagItem:
        """
        Upload and process a document for RAG.

        Args:
            user_id: User's Telegram ID
            filename: Original filename
            content: File content bytes
            scope: Scope (user, global)
            source_url: Optional source URL

        Returns:
            Created RagItem instance

        Raises:
            RAGError: If file type not supported or processing fails
        """
        # Validate file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise RAGError(
                f"Nieobsługiwany typ pliku: {file_ext}. Obsługiwane: {', '.join(self.SUPPORTED_EXTENSIONS)}",
                {"filename": filename, "extension": file_ext},
            )

        # Generate storage path
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        stored_filename = f"{user_id}_{file_hash}_{filename}"
        stored_path = os.path.join(self.storage_path, stored_filename)

        try:
            # Save file
            async with aiofiles.open(stored_path, "wb") as f:
                await f.write(content)

            # Extract text
            text = await self._extract_text(stored_path, file_ext)

            # Chunk text
            chunks = self._chunk_text(text)

            # Create RAG item
            rag_item = RagItem(
                user_id=user_id,
                scope=scope,
                source_type="upload",
                source_url=source_url,
                filename=filename,
                stored_path=stored_path,
                chunk_count=len(chunks),
                status="ready",
                item_metadata={
                    "file_size": len(content),
                    "file_hash": file_hash,
                    "chunks_preview": chunks[:3] if len(chunks) > 3 else chunks,
                },
            )

            self.db.add(rag_item)
            await self.db.flush()
            await self.db.refresh(rag_item)

            logger.info(f"Uploaded RAG document: {filename} ({len(chunks)} chunks)")

            return rag_item

        except Exception as e:
            logger.error(f"RAG upload error: {e}", exc_info=True)
            # Clean up file if exists
            if os.path.exists(stored_path):
                os.remove(stored_path)
            raise RAGError(
                f"Błąd przetwarzania dokumentu: {str(e)}",
                {"filename": filename},
            )

    async def search(
        self, user_id: int, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search RAG documents for relevant chunks.

        Simple keyword-based search (placeholder for vector search).

        Args:
            user_id: User's Telegram ID
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunk dicts
        """
        # Get user's RAG items
        result = await self.db.execute(
            select(RagItem)
            .where(RagItem.user_id == user_id, RagItem.status == "ready")
            .order_by(RagItem.created_at.desc())
        )
        rag_items = list(result.scalars().all())

        if not rag_items:
            return []

        # Simple keyword search (placeholder)
        # In production, use vector embeddings + similarity search
        query_lower = query.lower()
        results = []

        for item in rag_items:
            # Get chunks from metadata
            chunks = item.item_metadata.get("chunks_preview", [])

            for i, chunk in enumerate(chunks):
                # Simple keyword matching
                if any(word in chunk.lower() for word in query_lower.split()):
                    results.append(
                        {
                            "filename": item.filename,
                            "chunk_index": i,
                            "content": chunk,
                            "source_url": item.source_url,
                            "relevance_score": 0.5,  # Placeholder
                        }
                    )

        # Sort by relevance (placeholder - random for now)
        results = results[:top_k]

        return results

    async def list_documents(self, user_id: int) -> list[RagItem]:
        """
        List user's RAG documents.

        Args:
            user_id: User's Telegram ID

        Returns:
            List of RagItem instances
        """
        result = await self.db.execute(
            select(RagItem)
            .where(RagItem.user_id == user_id)
            .order_by(RagItem.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete_document(self, user_id: int, item_id: int) -> bool:
        """
        Delete a RAG document.

        Args:
            user_id: User's Telegram ID
            item_id: RagItem ID

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            select(RagItem).where(RagItem.id == item_id, RagItem.user_id == user_id)
        )
        rag_item = result.scalar_one_or_none()

        if not rag_item:
            return False

        # Delete file
        if os.path.exists(rag_item.stored_path):
            os.remove(rag_item.stored_path)

        # Delete DB record
        await self.db.delete(rag_item)
        await self.db.flush()

        return True

    async def _extract_text(self, file_path: str, file_ext: str) -> str:
        """
        Extract text from file.

        Args:
            file_path: Path to file
            file_ext: File extension

        Returns:
            Extracted text
        """
        if file_ext in {".txt", ".md", ".html", ".json"}:
            # Plain text files
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                return await f.read()

        elif file_ext == ".pdf":
            # PDF extraction (placeholder - requires pypdf or similar)
            # For now, return placeholder
            return "[PDF content extraction not implemented - install pypdf]"

        elif file_ext == ".docx":
            # DOCX extraction (placeholder - requires python-docx)
            return "[DOCX content extraction not implemented - install python-docx]"

        else:
            raise RAGError(f"Unsupported file type: {file_ext}")

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > 0:
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - self.CHUNK_OVERLAP

        return [c for c in chunks if len(c) > 50]  # Filter very short chunks
