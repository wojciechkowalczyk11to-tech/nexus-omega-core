"""
Embedding Service — generowanie osadzeń wektorowych dla RAG.

Wykorzystuje sentence-transformers z modelem all-MiniLM-L6-v2:
- Rozmiar: ~80MB
- Wymiary: 384
- Szybkość: ~2000 zdań/sekundę na CPU
- Jakość: doskonała dla semantic search

Model jest ładowany raz i cache'owany w pamięci.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from sentence_transformers import SentenceTransformer

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating vector embeddings using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 model:
    - Fast: ~2000 sentences/second on CPU
    - Compact: 384 dimensions
    - Quality: excellent for semantic search
    """

    _model: SentenceTransformer | None = None
    _model_name = "sentence-transformers/all-MiniLM-L6-v2"
    _dimensions = 384

    @classmethod
    def _load_model(cls) -> SentenceTransformer:
        """
        Lazy-load the embedding model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        if cls._model is None:
            logger.info(f"Loading embedding model: {cls._model_name}")
            cls._model = SentenceTransformer(cls._model_name)
            logger.info(f"Embedding model loaded successfully (dims={cls._dimensions})")
        return cls._model

    @classmethod
    async def generate_embedding(cls, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            384-dimensional embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * cls._dimensions

        model = cls._load_model()
        
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True).tolist(),
        )
        
        return embedding

    @classmethod
    async def generate_embeddings_batch(
        cls,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = cls._load_model()
        
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist(),
        )
        
        return embeddings

    @classmethod
    def get_dimensions(cls) -> int:
        """
        Get the dimensionality of embeddings.
        
        Returns:
            Number of dimensions (384)
        """
        return cls._dimensions

    @classmethod
    async def compute_similarity(
        cls,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)


# Convenience functions

async def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return await EmbeddingService.generate_embedding(text)


async def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    return await EmbeddingService.generate_embeddings_batch(texts, batch_size)


def get_embedding_dimensions() -> int:
    """Get the dimensionality of embeddings."""
    return EmbeddingService.get_dimensions()
