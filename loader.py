"""Document loader module supporting txt, pdf, and md files.

For digital PDFs: uses PyMuPDF (fitz) for text + table extraction.
For scanned PDFs: falls back to PaddleOCR PPStructureV3.
Automatically detects which approach to use per document.
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
        pages=[content] if content else [],
    )


# ------------------------------------------------------------------ #
#  Digital vs scanned PDF detection                                   #
# ------------------------------------------------------------------ #


def _is_digital_pdf(pdf_path: Path) -> bool:
    """Check if a PDF has embedded text (digital) or is image-only (scanned).

    A page is considered digital if it contains a meaningful amount of
    extractable text (>50 chars). If at least one page is digital, the
    whole document is treated as digital.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    for page in doc:
        text = page.get_text().strip()
        if len(text) > 50:
            doc.close()
            return True
    doc.close()
    return False


# ------------------------------------------------------------------ #
#  Digital PDF extraction with PyMuPDF                                #
# ------------------------------------------------------------------ #


def _extract_digital_pdf(pdf_path: Path) -> tuple:
    """Extract text and tables from a digital PDF using PyMuPDF.

    Returns:
        Tuple of (full_markdown: str, pages: List[str]) where pages is the
        per-page text for segmentation.
    """
    import fitz

    doc = fitz.open(str(pdf_path))

    # First pass: collect font sizes across all pages to determine heading thresholds
    font_sizes = _collect_font_sizes(doc)

    # Determine heading thresholds from font size distribution
    heading_thresholds = _compute_heading_thresholds(font_sizes)

    # Second pass: extract all blocks with metadata to detect repeating headers
    all_blocks_by_page: List[List[dict]] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_blocks = _extract_page_blocks(page, heading_thresholds)
        all_blocks_by_page.append(page_blocks)

    # Detect and remove repeating page headers/footers
    skip_texts = _detect_page_headers_footers(all_blocks_by_page, len(doc))

    # Build final markdown, skipping detected headers/footers
    markdown_pages: List[str] = []
    for page_blocks in all_blocks_by_page:
        page_parts: List[str] = []
        for block in page_blocks:
            if block["raw_text"] in skip_texts:
                continue
            page_parts.append(block["text"])
        if page_parts:
            markdown_pages.append("\n\n".join(page_parts))

    doc.close()
    full_text = "\n\n".join(markdown_pages)
    return full_text, markdown_pages


def _collect_font_sizes(doc) -> List[float]:
    """Collect all font sizes used across the document."""
    font_sizes: List[float] = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:  # skip image blocks
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span.get("size", 0), 1)
                    if size > 0:
                        font_sizes.append(size)
    return font_sizes


def _compute_heading_thresholds(font_sizes: List[float]) -> dict:
    """Compute h1/h2/h3 font-size thresholds from the document's font distribution.

    Uses the most common (body) font size as baseline. Anything larger becomes a heading.
    """
    if not font_sizes:
        return {"body": 10.0, "h3": 12.0, "h2": 14.0, "h1": 18.0}

    from collections import Counter
    size_counts = Counter(font_sizes)
    body_size = size_counts.most_common(1)[0][0]

    # Find sizes larger than body, sorted descending
    larger_sizes = sorted(
        [s for s in set(font_sizes) if s > body_size],
        reverse=True,
    )

    if len(larger_sizes) >= 3:
        return {
            "body": body_size,
            "h3": larger_sizes[2],
            "h2": larger_sizes[1],
            "h1": larger_sizes[0],
        }
    elif len(larger_sizes) == 2:
        return {
            "body": body_size,
            "h3": larger_sizes[1],
            "h2": larger_sizes[0],
            "h1": larger_sizes[0],
        }
    elif len(larger_sizes) == 1:
        return {
            "body": body_size,
            "h3": larger_sizes[0],
            "h2": larger_sizes[0],
            "h1": larger_sizes[0],
        }
    else:
        return {
            "body": body_size,
            "h3": body_size,
            "h2": body_size,
            "h1": body_size,
        }


def _extract_page_blocks(page, heading_thresholds: dict) -> List[dict]:
    """Extract a single page as a list of blocks with metadata.

    Each block: {"text": str, "raw_text": str, "y": float, "is_heading": bool}
    The raw_text is used for header/footer detection (no markdown prefix).
    """
    import fitz

    blocks_out: List[dict] = []

    # Get table bounding boxes so we can skip text that overlaps tables
    table_rects = []
    try:
        tables = page.find_tables()
        for table in tables:
            table_rects.append(table.bbox)
    except Exception:
        tables = []

    # Extract table content as HTML
    for table in tables:
        rows = table.extract()
        if rows:
            table_md = _table_to_html(rows)
            blocks_out.append({
                "text": table_md,
                "raw_text": table_md,
                "y": table.bbox[1],
                "is_heading": False,
            })

    # Build a set of rects that belong to tables (for text skip)
    table_rect_set = [fitz.Rect(r) for r in table_rects]

    # Extract text blocks with font info
    raw_blocks = page.get_text("dict")["blocks"]

    for block in raw_blocks:
        if block["type"] != 0:  # skip image blocks
            continue

        block_rect = fitz.Rect(block["bbox"])

        # Skip text blocks that overlap with tables
        overlaps_table = False
        for trect in table_rect_set:
            if block_rect.intersects(trect):
                overlaps_table = True
                break
        if overlaps_table:
            continue

        # Process lines in the block
        block_lines: List[str] = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    line_text += text + " "
            line_text = line_text.strip()
            if line_text:
                block_lines.append(line_text)

        if block_lines:
            block_text = " ".join(block_lines)
            max_font = max(
                (
                    span.get("size", 0)
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ),
                default=heading_thresholds["body"],
            )

            # Check if the block is short enough to be a heading
            is_short = len(block_text) < 200
            is_heading = False

            if is_short and max_font >= heading_thresholds["h1"]:
                block_md = f"# {block_text}"
                is_heading = True
            elif is_short and max_font >= heading_thresholds["h2"]:
                block_md = f"## {block_text}"
                is_heading = True
            elif is_short and max_font >= heading_thresholds["h3"]:
                block_md = f"### {block_text}"
                is_heading = True
            else:
                block_md = block_text

            blocks_out.append({
                "text": block_md,
                "raw_text": block_text,
                "y": block["bbox"][1],
                "is_heading": is_heading,
            })

    # Sort by vertical position
    blocks_out.sort(key=lambda b: b["y"])

    # Merge single-character headings into the next block (drop-cap artifacts)
    merged: List[dict] = []
    for i, block in enumerate(blocks_out):
        raw = block["raw_text"].strip()
        if (
            block["is_heading"]
            and len(raw) <= 2
            and i + 1 < len(blocks_out)
        ):
            # Drop-cap artifact (e.g. "T" from "This..."): merge into next block
            next_block = blocks_out[i + 1]
            merged_text = raw + next_block["raw_text"].lstrip()
            next_block["raw_text"] = merged_text
            next_block["text"] = merged_text  # no heading prefix
            next_block["is_heading"] = False
            continue  # skip the single-char heading
        merged.append(block)

    return merged


def _detect_page_headers_footers(
    all_blocks_by_page: List[List[dict]],
    total_pages: int,
) -> set:
    """Detect repeating page headers and footers across the document.

    A block is considered a header/footer if its normalized text appears
    on at least 30% of pages. Page numbers are normalized to a placeholder
    before comparison.
    """
    import re

    if total_pages < 3:
        return set()

    # Collect all block texts per page, normalized
    page_texts: List[List[str]] = []
    for page_blocks in all_blocks_by_page:
        texts = []
        for block in page_blocks:
            raw = block["raw_text"].strip()
            if not raw:
                continue
            # Normalize: lowercase, collapse whitespace, replace numbers with #
            norm = re.sub(r"\d+", "#", raw.lower())
            # Also normalize roman numerals (i, ii, iii, iv, v, vi, vii, viii, ix, x, etc.)
            norm = re.sub(
                r"\b[ivxlcdm]{1,6}\b",
                "#",
                norm,
            )
            norm = re.sub(r"\s+", " ", norm).strip()
            # Only consider short blocks as potential headers/footers
            if len(norm) < 200:
                texts.append(norm)
        page_texts.append(texts)

    # Count how many pages each normalized text appears on
    from collections import Counter
    text_page_counts: Counter = Counter()
    for page_texts_set in page_texts:
        for text in set(page_texts_set):  # deduplicate within page
            text_page_counts[text] += 1

    # A text is a header/footer if it appears on >= 30% of pages
    threshold = max(3, total_pages * 0.3)
    skip_norms = {text for text, count in text_page_counts.items() if count >= threshold}

    # Now find the original raw_text strings that match these normalized patterns
    skip_texts: set = set()
    for page_blocks in all_blocks_by_page:
        for block in page_blocks:
            raw = block["raw_text"].strip()
            if not raw:
                continue
            norm = re.sub(r"\d+", "#", raw.lower())
            norm = re.sub(r"\b[ivxlcdm]{1,6}\b", "#", norm)
            norm = re.sub(r"\s+", " ", norm).strip()
            if norm in skip_norms:
                skip_texts.add(raw)

    return skip_texts


def _table_to_html(rows: List[List[str]]) -> str:
    """Convert extracted table rows to an HTML table."""
    if not rows:
        return ""

    lines = ["<table border=\"1\">"]

    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        cells = []
        for cell in row:
            cell_text = str(cell).strip() if cell else ""
            cells.append(f"<{tag}>{cell_text}</{tag}>")
        lines.append("<tr>" + "".join(cells) + "</tr>")

    lines.append("</table>")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Scanned PDF extraction with PaddleOCR                              #
# ------------------------------------------------------------------ #


def _extract_scanned_pdf(
    pdf_path: Path,
    paddleocr_lang: str = "en",
    use_table_recognition: bool = True,
) -> str:
    """Extract text from a scanned/image PDF using PaddleOCR PPStructureV3."""
    try:
        from paddleocr import PPStructureV3
    except ImportError:
        raise ImportError(
            "PaddleOCR is required for scanned PDF loading. "
            "Install with: pip install paddleocr paddlepaddle"
        )

    import paddle

    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    pipeline = PPStructureV3(
        lang=paddleocr_lang,
        use_table_recognition=use_table_recognition,
        device=device,
    )

    output = pipeline.predict(input=str(pdf_path))

    markdown_list = []
    num_pages = 0
    for res in output:
        num_pages += 1
        md_info = res.markdown
        markdown_list.append(md_info)

    md_result = pipeline.concatenate_markdown_pages(markdown_list)
    return md_result["markdown_texts"]


# ------------------------------------------------------------------ #
#  PDF loader dispatcher                                              #
# ------------------------------------------------------------------ #


def _load_pdf_file(
    file_path: Path,
    paddleocr_lang: str = "en",
    use_table_recognition: bool = True,
    pdf_remove_patterns: Optional[List[str]] = None,
    parsed_cache_dir: Optional[str] = None,
) -> Document:
    """Load a PDF file. Automatically detects digital vs scanned.

    Digital PDFs → PyMuPDF (clean text + tables + font-based headings)
    Scanned PDFs → PaddleOCR PPStructureV3
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
                pages=[],  # per-page text not available from cache
            )

    # Detect if digital or scanned
    is_digital = _is_digital_pdf(file_path)
    pdf_method = "pymupdf" if is_digital else "paddleocr"

    pages: List[str] = []
    if is_digital:
        content, pages = _extract_digital_pdf(file_path)
        num_pages = _count_pdf_pages(file_path)
    else:
        content = _extract_scanned_pdf(
            file_path,
            paddleocr_lang=paddleocr_lang,
            use_table_recognition=use_table_recognition,
        )
        num_pages = len(content)  # approximate

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
            "pdf_method": pdf_method,
        },
        pages=pages,
    )


def _count_pdf_pages(pdf_path: Path) -> int:
    """Count the number of pages in a PDF."""
    import fitz
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
