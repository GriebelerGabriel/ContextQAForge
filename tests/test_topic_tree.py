"""Tests for the document-grounded topic tree module."""

import json

from topic_tree import DocumentTopicTree
from models import Document, SectionNode
from config import PipelineConfig


class TestDocumentTopicTree:
    def test_parse_document_headings(self, mock_config):
        """Test heading hierarchy extraction from Markdown."""
        tree = DocumentTopicTree(mock_config)
        doc = Document(
            content="# Title\n\nIntro text about the document.\n\n## Section A\n\nThis is the content for section A with enough text to pass filters.\n\n## Section B\n\nThis is the content for section B with enough text to pass filters.",
            source="test.md",
            doc_type="md",
        )

        tree.build_from_documents([doc])

        assert tree.root is not None
        leaves = tree.get_all_leaf_nodes()
        assert len(leaves) >= 2

    def test_parse_nested_headings(self, mock_config):
        """Test deeply nested heading structure."""
        tree = DocumentTopicTree(mock_config)
        doc = Document(
            content="# H1\n\nLevel one content with enough text to pass the minimum filter threshold.\n\n## H2\n\nLevel two content with enough text to pass the minimum filter threshold.\n\n### H3\n\nLevel three content with enough text to pass the minimum filter threshold.",
            source="test.md",
            doc_type="md",
        )

        tree.build_from_documents([doc])

        leaves = tree.get_all_leaf_nodes()
        assert len(leaves) >= 1

    def test_no_headings_creates_single_node(self, mock_config):
        """Test that documents without headings become a single node."""
        tree = DocumentTopicTree(mock_config)
        doc = Document(
            content="Just plain text without any headings. " * 10,
            source="plain.txt",
            doc_type="txt",
        )

        tree.build_from_documents([doc])

        leaves = tree.get_all_leaf_nodes()
        assert len(leaves) == 1
        assert leaves[0].content is not None

    def test_multiple_documents(self, mock_config):
        """Test tree built from multiple documents."""
        tree = DocumentTopicTree(mock_config)
        docs = [
            Document(content="# Doc1\n\nContent 1.", source="doc1.md", doc_type="md"),
            Document(content="# Doc2\n\nContent 2.", source="doc2.md", doc_type="md"),
        ]

        tree.build_from_documents(docs)

        leaves = tree.get_all_leaf_nodes()
        assert len(leaves) >= 2
        sources = {l.source for l in leaves}
        assert "doc1.md" in sources
        assert "doc2.md" in sources

    def test_chunk_leaves(self, mock_config):
        """Test chunking of leaf node content."""
        tree = DocumentTopicTree(mock_config)
        doc = Document(
            content="# Title\n\n" + "Word " * 200,
            source="test.md",
            doc_type="md",
        )

        tree.build_from_documents([doc])

        tree.chunk_leaves(chunk_size=100, chunk_overlap=20)

        for leaf in tree.get_all_leaf_nodes():
            if leaf.content:
                assert len(leaf.chunks) >= 1

    def test_qa_distribution_even(self, mock_config):
        """Test even distribution of QA pairs across leaves."""
        tree = DocumentTopicTree(mock_config)

        # Manually build a tree with 3 leaves
        tree.root = SectionNode(
            name="Root", source="root", depth=0,
            children=[
                SectionNode(name="A", source="a.md", depth=1, content="Content A"),
                SectionNode(name="B", source="b.md", depth=1, content="Content B"),
                SectionNode(name="C", source="c.md", depth=1, content="Content C"),
            ]
        )

        dist = tree.get_qa_distribution(9)
        assert len(dist) == 3
        assert all(count == 3 for _, count in dist)

    def test_qa_distribution_with_remainder(self, mock_config):
        """Test distribution when num_samples doesn't divide evenly."""
        tree = DocumentTopicTree(mock_config)

        tree.root = SectionNode(
            name="Root", source="root", depth=0,
            children=[
                SectionNode(name="A", source="a.md", depth=1, content="Content A"),
                SectionNode(name="B", source="b.md", depth=1, content="Content B"),
                SectionNode(name="C", source="c.md", depth=1, content="Content C"),
            ]
        )

        dist = tree.get_qa_distribution(10)
        total = sum(count for _, count in dist)
        assert total == 10
        # First leaf gets extra
        assert dist[0][1] == 4
        assert dist[1][1] == 3
        assert dist[2][1] == 3

    def test_get_leaf_path(self, mock_config):
        """Test path retrieval from root to leaf."""
        tree = DocumentTopicTree(mock_config)

        leaf = SectionNode(name="Grandchild", source="a.md", depth=2, content="Content")
        child = SectionNode(name="Child", source="a.md", depth=1, children=[leaf])
        tree.root = SectionNode(name="Root", source="root", depth=0, children=[child])

        path = tree.get_leaf_path(leaf)
        assert path == ["Root", "Child", "Grandchild"]

    def test_visualize(self, mock_config):
        """Test tree visualization output."""
        tree = DocumentTopicTree(mock_config)

        tree.root = SectionNode(
            name="Root", source="root", depth=0,
            children=[
                SectionNode(name="Child A", source="a.md", depth=1, content="Content A"),
                SectionNode(name="Child B", source="b.md", depth=1, content="Content B"),
            ]
        )
        tree.chunk_leaves()

        viz = tree.visualize()
        assert "Root" in viz
        assert "Child A" in viz
        assert "Child B" in viz
        assert "Leaf nodes: 2" in viz

    def test_visualize_no_tree(self, mock_config):
        """Test visualization with no tree built."""
        tree = DocumentTopicTree(mock_config)
        assert "No tree built" in tree.visualize()

    def test_save_and_load(self, mock_config, tmp_path):
        """Test JSON save/load round-trip."""
        tree = DocumentTopicTree(mock_config)

        tree.root = SectionNode(
            name="Root", source="root", depth=0,
            children=[
                SectionNode(name="Child", source="a.md", depth=1, content="Content", chunks=["Chunk1"]),
            ]
        )

        path = str(tmp_path / "tree.json")
        tree.save(path)

        tree2 = DocumentTopicTree(mock_config)
        tree2.load(path)

        assert tree2.root is not None
        assert tree2.root.name == "Root"
        assert tree2.root.children[0].name == "Child"
        assert tree2.root.children[0].chunks == ["Chunk1"]

    def test_split_text(self):
        """Test the text splitting utility."""
        text = "Word " * 200  # ~1000 chars
        chunks = DocumentTopicTree._split_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        # All chunks should be <= chunk_size (with small margin for boundary)
        for chunk in chunks:
            assert len(chunk) <= 250

    def test_split_text_short(self):
        """Test splitting text shorter than chunk_size."""
        text = "Short text."
        chunks = DocumentTopicTree._split_text(text, chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."
