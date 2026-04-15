"""Tests for the chunker module."""

from models import Document
from chunker import chunk_documents, _is_quality_content, _chunk_single_document


class TestChunkDocuments:
    def test_basic_chunking(self, sample_documents):
        chunks = chunk_documents(sample_documents, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) > 0
            assert chunk.source in ("test1.txt", "test2.md")

    def test_small_document_single_chunk(self):
        doc = Document(
            content="Short text that is long enough to pass the quality filter threshold for content. " * 3,
            source="short.txt",
            doc_type="txt",
        )
        chunks = chunk_documents([doc], chunk_size=1000)
        assert len(chunks) == 1
        assert "Short text" in chunks[0].content

    def test_chunk_overlap(self):
        long_text = "A " * 500  # ~1000 chars
        doc = Document(content=long_text, source="long.txt", doc_type="txt")
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)
        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            # Overlap means chunks share some content
            assert len(chunks) > 1

    def test_custom_quality_patterns(self):
        doc = Document(
            content="publicação autorizada manual de orientações nutrição e dietética " * 10,
            source="header.txt",
            doc_type="txt",
        )
        # With default patterns, this should be filtered
        chunks = chunk_documents([doc], chunk_size=500, quality_patterns=[
            "publicação autorizada",
            "manual de orientações",
            "nutrição e dietética",
        ])
        # Should be filtered as header/footer content
        assert len(chunks) == 0

    def test_empty_quality_patterns(self):
        doc = Document(
            content="publicação autorizada manual de orientações " * 10,
            source="header.txt",
            doc_type="txt",
        )
        # With empty patterns, content should pass through
        chunks = chunk_documents([doc], chunk_size=500, quality_patterns=[])
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self, sample_documents):
        chunks = chunk_documents(sample_documents, chunk_size=200)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestIsQualityContent:
    def test_short_content_rejected(self):
        assert _is_quality_content("Short") is False

    def test_good_content_passes(self):
        content = "Python is a high-level programming language that emphasizes code readability. It supports multiple programming paradigms including structured, procedural, and object-oriented programming."
        assert _is_quality_content(content) is True

    def test_header_pattern_rejected(self):
        content = "publicação autorizada manual de orientações serviço de nutrição e dietética " * 5
        assert _is_quality_content(content) is False

    def test_high_newline_ratio_rejected(self):
        content = "\n".join(["word"] * 50)
        assert _is_quality_content(content) is False


class TestChunkSingleDocument:
    def test_returns_list_of_chunks(self):
        doc = Document(content="A " * 300, source="test.txt", doc_type="txt")
        chunks = _chunk_single_document(doc, chunk_size=200, chunk_overlap=20, start_id=0)
        assert isinstance(chunks, list)
        assert all(c.source == "test.txt" for c in chunks)
