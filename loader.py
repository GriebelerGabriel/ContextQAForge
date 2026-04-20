"""Document loader module supporting txt, pdf, and md files.

Uses PaddleOCR PPStructureV3 for PDF parsing with layout detection,
table recognition, and heading hierarchy extraction.
"""

from pathlib import Path
from typing import List, Optional

from models import Document


def load_documents(
    folder_path: str,
    paddleocr_lang: str = "en",
    use_table_recognition: bool = True,
    pdf_remove_patterns: Optional[List[str]] = None,
    parsed_cache_dir: Optional[str] = None,
) -> List[Document]:
    """
    Load all supported documents from a folder.

    Args:
        folder_path: Path to folder containing documents
        paddleocr_lang: Language for PaddleOCR ("en", "pt", "latin")
        use_table_recognition: Whether to enable table recognition in PaddleOCR
        pdf_remove_patterns: Patterns to filter from PDF headers/footers
        parsed_cache_dir: Directory to cache parsed Markdown (skip re-parsing)

    Returns:
        List of Document objects
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if pdf_remove_patterns is None:
        from config import DEFAULT_PDF_REMOVE_PATTERNS
        pdf_remove_patterns = list(DEFAULT_PDF_REMOVE_PATTERNS)

    documents = []
    supported_extensions = {".txt", ".pdf", ".md"}

    for file_path in sorted(folder.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                doc = _load_single_document(
                    file_path,
                    paddleocr_lang=paddleocr_lang,
                    use_table_recognition=use_table_recognition,
                    pdf_remove_patterns=pdf_remove_patterns,
                    parsed_cache_dir=parsed_cache_dir,
                )
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents


def _load_single_document(
    file_path: Path,
    paddleocr_lang: str = "en",
    use_table_recognition: bool = True,
    pdf_remove_patterns: Optional[List[str]] = None,
    parsed_cache_dir: Optional[str] = None,
) -> Document:
    """Load a single document based on its extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".txt" or suffix == ".md":
        return _load_text_file(file_path)
    elif suffix == ".pdf":
        return _load_pdf_file(
            file_path,
            paddleocr_lang=paddleocr_lang,
            use_table_recognition=use_table_recognition,
            pdf_remove_patterns=pdf_remove_patterns,
            parsed_cache_dir=parsed_cache_dir,
        )
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_text_file(file_path: Path) -> Document:
    """Load a text or markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    doc_type = "md" if file_path.suffix.lower() == ".md" else "txt"

    return Document(
        content=content,
        source=file_path.name,
        doc_type=doc_type,
        metadata={"filename": file_path.name, "size": len(content)},
    )


def _load_pdf_file(
    file_path: Path,
    paddleocr_lang: str = "en",
    use_table_recognition: bool = True,
    pdf_remove_patterns: Optional[List[str]] = None,
    parsed_cache_dir: Optional[str] = None,
) -> Document:
    """Load a PDF file using PaddleOCR PPStructureV3.

    Converts PDF pages to structured Markdown with heading hierarchy,
    tables (HTML), and layout-preserving paragraphs.
    """
    # Check for cached parsed output
    if parsed_cache_dir:
        cache_path = Path(parsed_cache_dir) / f"{file_path.stem}.md"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                content = f.read()
            return Document(
                content=content,
                source=file_path.name,
                doc_type="pdf",
                metadata={
                    "filename": file_path.name,
                    "size": len(content),
                    "cached": True,
                },
            )

    try:
        from paddleocr import PPStructureV3
    except ImportError:
        raise ImportError(
            "PaddleOCR is required for PDF loading. "
            "Install with: pip install paddleocr paddlepaddle"
        )

    # Initialize PPStructureV3 pipeline (use GPU if available)
    import paddle

    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    pipeline = PPStructureV3(
        lang=paddleocr_lang,
        use_table_recognition=use_table_recognition,
        device=device,
    )

    # Process PDF — predict yields results per page
    output = pipeline.predict(input=str(file_path))

    markdown_list = []
    num_pages = 0
    for res in output:
        num_pages += 1
        md_info = res.markdown
        markdown_list.append(md_info)

    # Concatenate all pages into a single Markdown document
    md_result = pipeline.concatenate_markdown_pages(markdown_list)
    content = md_result["markdown_texts"]

    # Ensure cache directory exists and save
    if parsed_cache_dir:
        cache_path = Path(parsed_cache_dir) / f"{file_path.stem}.md"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)

    return Document(
        content=content,
        source=file_path.name,
        doc_type="pdf",
        metadata={
            "filename": file_path.name,
            "size": len(content),
            "num_pages": num_pages,
        },
    )
