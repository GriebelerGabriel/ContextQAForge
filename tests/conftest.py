"""Shared test fixtures."""

import os
import pytest
from unittest.mock import MagicMock

from models import Chunk, Document, SectionNode
from config import PipelineConfig


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="# Introduction\n\nPython is a programming language.\n\n## Features\n\nPython has dynamic typing.\n\n## History\n\nPython was created in 1991.",
            source="test1.md",
            doc_type="md",
            metadata={"filename": "test1.md"},
        ),
        Document(
            content="# Machine Learning\n\nML is a subset of AI.\n\n## Deep Learning\n\nDeep learning uses neural networks.\n\n## Applications\n\nML is used in many fields.",
            source="test2.md",
            doc_type="md",
            metadata={"filename": "test2.md"},
        ),
    ]


@pytest.fixture
def sample_section_node():
    """Create a sample section node for testing."""
    return SectionNode(
        name="Test Section",
        source="test.md",
        depth=1,
        content="This is a test section with some content. " * 20,
    )


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
