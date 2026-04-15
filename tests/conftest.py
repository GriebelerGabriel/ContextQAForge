"""Shared test fixtures."""

import os
import pytest
from unittest.mock import MagicMock

from models import Chunk, Document
from config import PipelineConfig


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991. Python is known for its readability and simplicity.",
            source="test1.txt",
            doc_type="txt",
            metadata={"filename": "test1.txt"},
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence. It enables systems to learn from data without being explicitly programmed. Deep learning is a further subset of machine learning.",
            source="test2.md",
            doc_type="md",
            metadata={"filename": "test2.md"},
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(content="Python is a high-level programming language.", source="test.txt", chunk_id=0, start_pos=0, end_pos=45),
        Chunk(content="Machine learning is a subset of artificial intelligence.", source="test.txt", chunk_id=1, start_pos=46, end_pos=100),
        Chunk(content="Deep learning uses neural networks with many layers.", source="test.txt", chunk_id=2, start_pos=101, end_pos=155),
    ]


@pytest.fixture
def mock_config():
    """Create a PipelineConfig with a fake API key."""
    return PipelineConfig(
        openai_api_key="test-key-12345",
        chunk_size=500,
        chunk_overlap=50,
        num_samples=10,
        model_name="gpt-4o-mini",
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI chat completion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = '{"question": "O que e Python?", "ground_truth": "Python e uma linguagem de programacao.", "type": "single-hop", "difficulty": "easy"}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response
