"""Tests for the loader module."""

import os
import tempfile
from pathlib import Path

from loader import load_documents, _clean_pdf_text


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

    def test_custom_remove_patterns(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Some content here about technology and programming.", encoding="utf-8")
        # Pass empty patterns to test parameter wiring
        docs = load_documents(str(tmp_path), pdf_remove_patterns=[])
        assert len(docs) == 1

    def test_recursive_loading(self, tmp_path):
        subfolder = tmp_path / "subdir"
        subfolder.mkdir()
        (subfolder / "nested.txt").write_text("Nested content " * 20, encoding="utf-8")
        (tmp_path / "root.txt").write_text("Root content " * 20, encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2


class TestCleanPdfText:
    def test_removes_configured_patterns(self):
        text = "Good content about health and nutrition.\nmanual de orientações\nMore good content about diet."
        result = _clean_pdf_text(text, remove_patterns=["manual de orientações"])
        assert "manual de orientações" not in result.lower()

    def test_empty_patterns_pass_through(self):
        text = "Some content " * 10
        result = _clean_pdf_text(text, remove_patterns=[])
        assert len(result) > 0

    def test_default_patterns_used_when_none(self):
        text = (
            "Python is a versatile programming language used for web development, "
            "data analysis, artificial intelligence, and scientific computing. "
            "It was created by Guido van Rossum and first released in 1991. "
            "Python's design philosophy emphasizes code readability with its notable "
            "use of significant indentation. Its language constructs and object-oriented "
            "approach aim to help programmers write clear, logical code for small and "
            "large-scale projects."
        )
        result = _clean_pdf_text(text, remove_patterns=None)
        assert len(result) > 0
