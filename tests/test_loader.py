"""Tests for the loader module."""

import os
import tempfile
from pathlib import Path

from loader import load_documents, _load_text_file


class TestLoadDocuments:
    def test_load_txt_file(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world, this is a test document with some content.", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].doc_type == "txt"
        assert "Hello world" in docs[0].content

    def test_load_md_file(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nSome markdown content here.", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].doc_type == "md"

    def test_unsupported_files_ignored(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 0

    def test_folder_not_found_raises(self):
        try:
            load_documents("/nonexistent/path")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_recursive_loading(self, tmp_path):
        subfolder = tmp_path / "subdir"
        subfolder.mkdir()
        (subfolder / "nested.txt").write_text("Nested content " * 20, encoding="utf-8")
        (tmp_path / "root.txt").write_text("Root content " * 20, encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2

    def test_pdf_cached_output_used(self, tmp_path):
        """Test that cached parsed Markdown is used when available."""
        # Put PDF in a subfolder so the cache .md file doesn't get loaded too
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        pdf_file = docs_dir / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")
        cached_md = cache_dir / "test.md"
        cached_md.write_text("# Cached Title\n\nCached content from previous parse.", encoding="utf-8")

        docs = load_documents(str(docs_dir), parsed_cache_dir=str(cache_dir))
        assert len(docs) == 1
        assert "Cached content" in docs[0].content
        assert docs[0].metadata.get("cached") is True

    def test_documents_sorted_by_name(self, tmp_path):
        """Test that documents are loaded in sorted order."""
        (tmp_path / "z_doc.txt").write_text("Z content", encoding="utf-8")
        (tmp_path / "a_doc.txt").write_text("A content", encoding="utf-8")
        (tmp_path / "m_doc.txt").write_text("M content", encoding="utf-8")

        docs = load_documents(str(tmp_path))
        assert len(docs) == 3
        # Should be sorted alphabetically
        assert docs[0].source.endswith("a_doc.txt")
        assert docs[2].source.endswith("z_doc.txt")
