"""FAISS vector store for efficient similarity search."""

from typing import List, Optional, Tuple

import faiss
import numpy as np

from chunker import Chunk


class VectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(self, dimension: int, metric: int = faiss.METRIC_L2):
        """
        Initialize vector store.

        Args:
            dimension: Dimension of embeddings
            metric: FAISS distance metric (default: L2)
        """
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.chunks: List[Chunk] = []
        self._is_trained = False

    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: Numpy array of shape (n_chunks, dimension)
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings")

        self.chunks = chunks

        # Normalize embeddings for cosine similarity if using inner product
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(embeddings.copy())

        # Create index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)

        # Add embeddings
        self.index.add(embeddings)
        self._is_trained = True

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5, exclude_indices: Optional[set] = None
    ) -> Tuple[List[Chunk], List[float]]:
        """
        Search for most similar chunks with diversity filtering.

        Retrieves 3x the requested results and filters out near-duplicate
        chunks (high text overlap) to ensure context diversity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            exclude_indices: Set of chunk indices to exclude from results

        Returns:
            Tuple of (chunks, distances/scores)
        """
        if not self._is_trained or self.index is None:
            raise RuntimeError("Vector store is empty. Add documents before searching.")

        if exclude_indices is None:
            exclude_indices = set()

        # Normalize query for cosine similarity
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Reshape for FAISS
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)

        # Over-retrieve to allow diversity filtering + exclusions
        fetch_k = min(top_k * 3 + len(exclude_indices), len(self.chunks))
        distances, indices = self.index.search(query_vector, fetch_k)

        # Retrieve chunks, excluding used ones
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.chunks) and idx not in exclude_indices:
                candidates.append((self.chunks[idx], float(dist), idx))

        # Diversity filter: skip chunks too similar to already-selected ones
        results = []
        scores = []
        result_indices = []
        for chunk, score, idx in candidates:
            if len(results) >= top_k:
                break
            if self._is_too_similar(chunk.content, [r.content for r in results]):
                continue
            results.append(chunk)
            scores.append(score)
            result_indices.append(idx)

        return results, scores, result_indices

    @staticmethod
    def _is_too_similar(text: str, existing_texts: List[str], threshold: float = 0.6) -> bool:
        """Check if text is too similar to any existing text via word overlap."""
        if not existing_texts:
            return False
        words = set(text.lower().split())
        if not words:
            return True
        for existing in existing_texts:
            existing_words = set(existing.lower().split())
            overlap = len(words & existing_words) / max(len(words), 1)
            if overlap > threshold:
                return True
        return False

    def get_random_chunks(self, n: int = 1) -> List[Chunk]:
        """Get n random chunks from the store."""
        if not self.chunks:
            return []
        import random
        return random.sample(self.chunks, min(n, len(self.chunks)))

    def __len__(self) -> int:
        """Return number of chunks in store."""
        return len(self.chunks)
