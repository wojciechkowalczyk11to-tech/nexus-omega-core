"""
RAG Tool V2 — pgvector-powered semantic search with embeddings.

Improvements over V1:
- Vector embeddings using sentence-transformers
- Semantic similarity search with pgvector
- Proper chunking strategy (semantic + overlap)
- Reranking for better relevance
- Separate chunk storage in database
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import aiofiles
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import RAGError
from app.core.logging_config import get_logger
from app.db.models.rag_chunk import RagChunk
from app.db.models.rag_item import RagItem
from app.services.embedding_service import embed_text, embed_texts

logger = get_logger(__name__)


class RAGToolV2:
    """
    Advanced RAG tool with vector embeddings and semantic search.

    Features:
    - pgvector for similarity search
    - sentence-transformers for embeddings
    - Semantic chunking with overlap
    - Reranking for better results
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".pdf",
        ".docx",
        ".html",
        ".json",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".go",
        ".rs",
    }

    # Chunking parameters
    CHUNK_SIZE = 800  # characters (optimized for semantic units)
    CHUNK_OVERLAP = 200  # characters
    MIN_CHUNK_SIZE = 100  # minimum chunk size

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
        Upload and process a document for RAG with vector embeddings.

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
            text_content = await self._extract_text(stored_path, file_ext)

            # Chunk text with semantic boundaries
            chunks = self._chunk_text_semantic(text_content)

            if not chunks:
                raise RAGError("Nie udało się wyekstrahować tekstu z dokumentu")

            # Create RAG item
            rag_item = RagItem(
                user_id=user_id,
                scope=scope,
                source_type="upload",
                source_url=source_url,
                filename=filename,
                stored_path=stored_path,
                chunk_count=len(chunks),
                status="processing",
                item_metadata={
                    "file_size": len(content),
                    "file_hash": file_hash,
                    "file_ext": file_ext,
                },
            )

            self.db.add(rag_item)
            await self.db.flush()
            await self.db.refresh(rag_item)

            # Generate embeddings for all chunks in batch
            logger.info("Generating embeddings for %s chunks...", len(chunks))
            embeddings = await embed_texts(chunks, batch_size=32)

            # Create chunk records with embeddings
            chunk_records = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
                chunk_record = RagChunk(
                    user_id=user_id,
                    rag_item_id=rag_item.id,
                    content=chunk_text,
                    chunk_index=i,
                    embedding=embedding,
                    chunk_metadata={
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                    },
                )
                chunk_records.append(chunk_record)

            self.db.add_all(chunk_records)

            # Update status
            rag_item.status = "indexed"
            await self.db.flush()

            logger.info("Uploaded and indexed RAG document: %s (%s chunks)", filename, len(chunks))

            return rag_item

        except Exception as e:
            logger.error("RAG upload error: %s", e, exc_info=True)
            # Clean up file if exists
            if os.path.exists(stored_path):
                os.remove(stored_path)
            raise RAGError(
                f"Błąd przetwarzania dokumentu: {str(e)}",
                {"filename": filename},
            ) from e

    async def search_semantic(
        self,
        user_id: int,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Search RAG documents using semantic similarity (pgvector).

        Args:
            user_id: User's Telegram ID
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of relevant chunk dicts with similarity scores
        """
        # Generate query embedding
        query_embedding = await embed_text(query)

        # Perform vector similarity search using pgvector
        # Using cosine distance operator (<=>)
        search_query = text("""
            SELECT
                rc.id,
                rc.content,
                rc.chunk_index,
                rc.chunk_metadata,
                ri.filename,
                ri.source_url,
                1 - (rc.embedding <=> :query_embedding) as similarity_score
            FROM rag_chunks rc
            JOIN rag_items ri ON rc.rag_item_id = ri.id
            WHERE rc.user_id = :user_id
            AND ri.status = 'indexed'
            AND 1 - (rc.embedding <=> :query_embedding) >= :threshold
            ORDER BY rc.embedding <=> :query_embedding
            LIMIT :limit
        """)

        result = await self.db.execute(
            search_query,
            {
                "user_id": user_id,
                "query_embedding": query_embedding,
                "threshold": similarity_threshold,
                "limit": top_k * 2,  # Get more for reranking
            },
        )

        rows = result.fetchall()

        if not rows:
            return []

        # Convert to dicts
        results = [
            {
                "chunk_id": row[0],
                "content": row[1],
                "chunk_index": row[2],
                "chunk_metadata": row[3],
                "filename": row[4],
                "source_url": row[5],
                "similarity_score": float(row[6]),
            }
            for row in rows
        ]

        # Simple reranking: prefer chunks with query keywords
        results = self._rerank_results(results, query)

        # Return top_k after reranking
        return results[:top_k]

    def _rerank_results(
        self,
        results: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Rerank results using simple keyword matching boost.

        Args:
            results: Initial search results
            query: Original query

        Returns:
            Reranked results
        """
        query_words = set(query.lower().split())

        for result in results:
            content_lower = result["content"].lower()

            # Count keyword matches
            keyword_matches = sum(1 for word in query_words if word in content_lower)

            # Boost score based on keyword matches
            keyword_boost = keyword_matches * 0.05  # 5% boost per keyword
            result["similarity_score"] = min(1.0, result["similarity_score"] + keyword_boost)
            result["keyword_matches"] = keyword_matches

        # Sort by boosted score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

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
            select(RagItem).where(RagItem.user_id == user_id).order_by(RagItem.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete_document(self, user_id: int, item_id: int) -> bool:
        """
        Delete a RAG document and all its chunks.

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

        # Delete DB record (chunks will be cascade deleted)
        await self.db.delete(rag_item)
        await self.db.flush()

        logger.info("Deleted RAG document: %s (id=%s)", rag_item.filename, item_id)

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
        if file_ext in {
            ".txt",
            ".md",
            ".html",
            ".json",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
        }:
            # Plain text files
            async with aiofiles.open(file_path, encoding="utf-8", errors="ignore") as f:
                return await f.read()

        elif file_ext == ".pdf":
            # PDF extraction using pypdf
            try:
                from pypdf import PdfReader

                reader = PdfReader(file_path)
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text())
                return "\n\n".join(text_parts)
            except Exception as e:
                logger.error("PDF extraction error: %s", e)
                raise RAGError(f"Błąd ekstrakcji PDF: {str(e)}") from e

        elif file_ext == ".docx":
            # DOCX extraction using python-docx
            try:
                from docx import Document

                doc = Document(file_path)
                text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
                return "\n\n".join(text_parts)
            except Exception as e:
                logger.error("DOCX extraction error: %s", e)
                raise RAGError(f"Błąd ekstrakcji DOCX: {str(e)}") from e

        else:
            raise RAGError(f"Nieobsługiwany typ pliku: {file_ext}")

    def _chunk_text_semantic(self, text: str) -> list[str]:
        """
        Chunk text with semantic boundaries (paragraphs, sentences).

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If paragraph is too long, split by sentences
            if len(para) > self.CHUNK_SIZE:
                # Split by sentences (simple heuristic)
                sentences = [s.strip() + "." for s in para.split(". ") if s.strip()]

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.CHUNK_SIZE:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            else:
                # Add paragraph to current chunk
                if len(current_chunk) + len(para) <= self.CHUNK_SIZE:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para

        # Add remaining chunk
        if current_chunk and len(current_chunk) >= self.MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())

        # Add overlap between chunks for better context
        chunks_with_overlap = []
        for i, chunk in enumerate(chunks):
            if i > 0 and len(chunks[i - 1]) > self.CHUNK_OVERLAP:
                # Add overlap from previous chunk
                overlap = chunks[i - 1][-self.CHUNK_OVERLAP :]
                chunk = overlap + " " + chunk
            chunks_with_overlap.append(chunk)

        return chunks_with_overlap
