"""Pydantic models for the RAG QA Pipeline."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document with content and metadata."""

    content: str = Field(..., description="Document text content (structured Markdown for PDFs)")
    source: str = Field(..., description="File path or source identifier")
    doc_type: Literal["txt", "pdf", "md", "unknown"] = Field(
        default="unknown", description="Document type"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Chunk(BaseModel):
    """A chunk of text with source reference."""

    content: str = Field(..., description="Chunk text content")
    source: str = Field(..., description="Source document identifier")
    chunk_id: int = Field(..., description="Unique chunk identifier")
    start_pos: int = Field(default=0, description="Start position in original document")
    end_pos: int = Field(default=0, description="End position in original document")


class SectionNode(BaseModel):
    """A node in the document-grounded topic tree."""

    name: str = Field(..., description="Section heading text")
    source: str = Field(..., description="Source document filename")
    depth: int = Field(default=0, description="Depth in tree (0 = root)")
    content: Optional[str] = Field(
        default=None, description="Raw section text (leaf nodes)"
    )
    chunks: List[str] = Field(
        default_factory=list, description="Smaller chunks after chunking step"
    )
    children: List["SectionNode"] = Field(
        default_factory=list, description="Child sections"
    )

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""
        return len(self.children) == 0

    @property
    def path(self) -> List[str]:
        """Get the topic path by traversing up (requires parent context)."""
        return [self.name]


class QAPair(BaseModel):
    """A question-answer pair with contexts and metadata."""

    question: str = Field(..., description="Generated question")
    ground_truth: str = Field(..., description="Ground truth answer")
    contexts: List[str] = Field(..., description="Retrieved context chunks")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including type, difficulty, and source"
    )


class QAType(BaseModel):
    """QA type specification."""

    type: Literal["single-hop", "multi-hop", "inference", "unanswerable", "paraphrase"] = Field(
        ..., description="Type of question"
    )
    description: str = Field(..., description="Description of this QA type")


class DatasetEntry(BaseModel):
    """Single dataset entry in RAGAS evaluation format.

    Required for RAGAS evaluation:
    - question: The test question
    - answer: The generated answer (same as ground_truth for synthetic datasets)
    - contexts: List of source text chunks that contain the answer
    - ground_truth: The expected correct answer
    """

    question: str
    answer: str
    contexts: List[str]  # Source chunks used to answer the question
    ground_truth: str
