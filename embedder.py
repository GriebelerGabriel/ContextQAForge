"""OpenAI embeddings module with batching support."""

from typing import Dict, List

import numpy as np
from openai import OpenAI

from config import PipelineConfig


class EmbeddingCache:
    """Caches embedding vectors to avoid redundant API calls."""

    def __init__(self, embedder: "Embedder"):
        self._embedder = embedder
        self._cache: Dict[str, np.ndarray] = {}

    def embed_query(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        key = text.strip().lower()
        if key not in self._cache:
            self._cache[key] = self._embedder.embed_query(text)
        return self._cache[key]

    def precompute(self, texts: List[str]) -> None:
        """Pre-compute embeddings for multiple texts in one batch call."""
        # Find uncached texts
        uncached = [t for t in texts if t.strip().lower() not in self._cache]
        if uncached:
            embeddings = self._embedder.embed_chunks(uncached)
            for text, embedding in zip(uncached, embeddings):
                self._cache[text.strip().lower()] = embedding


class Embedder:
    """Handles embedding generation using OpenAI API."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.embedding_model
        self.dimension: int = 0  # Set during first embed_chunks call

    def embed_chunks(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per batch

        Returns:
            Numpy array of shape (len(texts), embedding_dimension)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        if self.dimension == 0:
            self.dimension = embeddings_array.shape[1]

        return embeddings_array

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        # Clean empty strings
        cleaned_texts = [t if t.strip() else " " for t in texts]

        response = self.client.embeddings.create(
            model=self.model,
            input=cleaned_texts,
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        embeddings = self.embed_chunks([query], batch_size=1)
        return embeddings[0]
