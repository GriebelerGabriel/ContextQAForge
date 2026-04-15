"""Tests for the vector_store module."""

import numpy as np
import pytest

from vector_store import VectorStore
from models import Chunk


class TestVectorStore:
    def test_add_and_search(self):
        dim = 8
        store = VectorStore(dimension=dim)

        chunks = [
            Chunk(content="Python programming", source="test.txt", chunk_id=0, start_pos=0, end_pos=20),
            Chunk(content="Java programming", source="test.txt", chunk_id=1, start_pos=20, end_pos=40),
            Chunk(content="Cooking recipes", source="test.txt", chunk_id=2, start_pos=40, end_pos=60),
        ]

        # Create embeddings where similar content has similar vectors
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        store.add_documents(chunks, embeddings)
        assert len(store) == 3

        # Search with a query similar to first chunk
        query = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results, scores, indices = store.search(query, top_k=2)
        assert len(results) == 2
        assert results[0].content == "Python programming"

    def test_search_empty_store_raises(self):
        store = VectorStore(dimension=8)
        query = np.zeros(8, dtype=np.float32)
        try:
            store.search(query)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_mismatched_embeddings_raises(self):
        store = VectorStore(dimension=8)
        chunks = [
            Chunk(content="test", source="test.txt", chunk_id=0, start_pos=0, end_pos=4),
        ]
        embeddings = np.zeros((2, 8), dtype=np.float32)  # 2 embeddings for 1 chunk
        try:
            store.add_documents(chunks, embeddings)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_get_random_chunks(self):
        store = VectorStore(dimension=8)
        chunks = [
            Chunk(content=f"chunk {i}", source="test.txt", chunk_id=i, start_pos=0, end_pos=10)
            for i in range(5)
        ]
        embeddings = np.random.rand(5, 8).astype(np.float32)
        store.add_documents(chunks, embeddings)

        random_chunks = store.get_random_chunks(n=3)
        assert len(random_chunks) == 3

    def test_get_random_chunks_empty(self):
        store = VectorStore(dimension=8)
        assert store.get_random_chunks(n=3) == []

    def test_embeddings_not_mutated(self):
        """Verify that add_documents does not mutate the input embeddings array."""
        dim = 4
        store = VectorStore(dimension=dim, metric=0)  # METRIC_INNER_PRODUCT = 0

        chunks = [
            Chunk(content="test", source="t.txt", chunk_id=0, start_pos=0, end_pos=4),
        ]
        embeddings = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        original = embeddings.copy()

        store.add_documents(chunks, embeddings)

        # The original array should NOT be modified
        np.testing.assert_array_equal(embeddings, original)

    def test_len(self):
        store = VectorStore(dimension=8)
        assert len(store) == 0

        chunks = [
            Chunk(content="test", source="test.txt", chunk_id=0, start_pos=0, end_pos=4),
        ]
        embeddings = np.random.rand(1, 8).astype(np.float32)
        store.add_documents(chunks, embeddings)
        assert len(store) == 1

    def test_search_diversity_filters_near_duplicates(self):
        """Search should not return chunks with high text overlap."""
        dim = 8
        store = VectorStore(dimension=dim)

        chunks = [
            Chunk(
                content="Pode ser utilizada para fazer a salada de maionese com batatas, para servir como um molho para saladas.",
                source="test.txt", chunk_id=0, start_pos=0, end_pos=100,
            ),
            Chunk(
                content="Pode ser utilizada para fazer a salada de maionese com batatas, para servir como um molho.",
                source="test.txt", chunk_id=1, start_pos=100, end_pos=200,
            ),
            Chunk(
                content="As doencas cardiovasculares estao associadas a pior qualidade de vida e altos niveis de mortalidade.",
                source="test.txt", chunk_id=2, start_pos=200, end_pos=300,
            ),
        ]
        # First two chunks have similar embeddings, third is different
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        store.add_documents(chunks, embeddings)
        query = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results, scores, _ = store.search(query, top_k=2)

        # Should get one maionese chunk + the cardiovascular chunk (diverse)
        contents = [r.content for r in results]
        assert len(results) == 2
        # At least one result should be about cardiovascular (not both maionese)
        assert any("cardiovasculares" in c for c in contents)

    def test_is_too_similar(self):
        text1 = "Pode ser utilizada para fazer a salada de maionese com batatas"
        text2 = "Pode ser utilizada para fazer a salada de maionese com batatas e servir"
        assert VectorStore._is_too_similar(text2, [text1]) is True

    def test_is_not_too_similar(self):
        text1 = "Pode ser utilizada para fazer a salada de maionese com batatas"
        text2 = "As doencas cardiovasculares estao associadas a pior qualidade de vida"
        assert VectorStore._is_too_similar(text2, [text1]) is False
