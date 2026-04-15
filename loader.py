"""Document loader module supporting txt, pdf, and md files."""

import json
import os
from pathlib import Path
from typing import List, Optional

from models import Document


def load_documents(
    folder_path: str,
    use_llm_for_pdf: bool = False,
    pdf_extraction_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    pdf_remove_patterns: Optional[List[str]] = None,
) -> List[Document]:
    """
    Load all supported documents from a folder.

    Args:
        folder_path: Path to folder containing documents
        use_llm_for_pdf: Whether to use LLM for PDF table/structure extraction
        pdf_extraction_model: Model to use for PDF extraction
        api_key: OpenAI API key for LLM extraction
        pdf_remove_patterns: Patterns to filter from PDF headers/footers

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

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                doc = _load_single_document(
                    file_path,
                    use_llm_for_pdf=use_llm_for_pdf,
                    pdf_extraction_model=pdf_extraction_model,
                    api_key=api_key,
                    pdf_remove_patterns=pdf_remove_patterns,
                )
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents


def _load_single_document(
    file_path: Path,
    use_llm_for_pdf: bool = False,
    pdf_extraction_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    pdf_remove_patterns: Optional[List[str]] = None,
) -> Document:
    """Load a single document based on its extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".txt" or suffix == ".md":
        return _load_text_file(file_path)
    elif suffix == ".pdf":
        return _load_pdf_file(
            file_path, use_llm_for_pdf, pdf_extraction_model, api_key, pdf_remove_patterns
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
        source=str(file_path),
        doc_type=doc_type,
        metadata={"filename": file_path.name, "size": len(content)},
    )


def _load_pdf_file(
    file_path: Path,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    pdf_remove_patterns: Optional[List[str]] = None,
) -> Document:
    """Load a PDF file using pdfplumber with automatic header/footer removal."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF loading. Install with: pip install pdfplumber")

    all_text = []
    num_pages = 0

    with pdfplumber.open(str(file_path)) as pdf:
        num_pages = len(pdf.pages)

        for page in pdf.pages:
            # Get page dimensions
            width = page.width
            height = page.height

            # Define crop box to exclude headers (top 10%) and footers (bottom 10%)
            # This removes repetitive page headers/footers
            crop_box = (0, height * 0.10, width, height * 0.90)

            # Extract text from cropped area
            cropped_page = page.crop(crop_box)
            page_text = cropped_page.extract_text()

            if page_text:
                all_text.append(page_text)

    raw_content = "\n\n".join(all_text)

    # Clean the text to remove any remaining noise
    raw_content = _clean_pdf_text(raw_content, pdf_remove_patterns)

    # If LLM extraction enabled, send to OpenAI for structured extraction
    if use_llm and api_key:
        content = _extract_pdf_with_llm(raw_content, file_path.name, llm_model, api_key)
    else:
        content = raw_content

    return Document(
        content=content,
        source=str(file_path),
        doc_type="pdf",
        metadata={
            "filename": file_path.name,
            "num_pages": num_pages,
            "size": len(content),
            "llm_extracted": use_llm and api_key is not None,
        },
    )


def _clean_pdf_text(text: str, remove_patterns: Optional[List[str]] = None) -> str:
    """Clean remaining noise from PDF text after cropping."""
    import re

    if remove_patterns is None:
        from config import DEFAULT_PDF_REMOVE_PATTERNS
        remove_patterns = list(DEFAULT_PDF_REMOVE_PATTERNS)

    # First, normalize all whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove table of contents patterns (lines with numbers followed by titles)
    text = re.sub(r'[^\n]*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\n]*\?\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    lines = text.split("\n")
    cleaned = []
    
    for line in lines:
        stripped = line.strip()
        line_lower = stripped.lower()

        # Skip empty or very short lines
        if not stripped or len(stripped) < 20:
            continue

        # Skip lines that are just numbers or mostly numbers
        if stripped.isdigit() or sum(c.isdigit() for c in stripped) > len(stripped) * 0.5:
            continue

        # Skip lines matching header patterns
        skip = False
        for pattern in remove_patterns:
            if pattern in line_lower:
                skip = True
                break

        if not skip:
            cleaned.append(stripped)

    # Join ALL lines with spaces (fix broken lines from PDF extraction)
    # Then split by sentence-ending punctuation to create proper paragraphs
    full_text = " ".join(cleaned)
    
    # Split into sentences at punctuation
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    # Group sentences into paragraphs (every 3-5 sentences or at natural breaks)
    paragraphs = []
    current_para = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        current_para.append(sentence)
        
        # Start new paragraph after 3-5 sentences or if sentence is a heading
        if len(current_para) >= 4 or (len(sentence) < 50 and not sentence.endswith(',')):
            paragraphs.append(" ".join(current_para))
            current_para = []
    
    if current_para:
        paragraphs.append(" ".join(current_para))
    
    # Filter out very short paragraphs
    paragraphs = [p for p in paragraphs if len(p) > 100]
    
    return "\n\n".join(paragraphs)


def _extract_pdf_with_llm(raw_text: str, filename: str, model: str, api_key: str) -> str:
    """Use LLM to extract structured content from PDF, including tables."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Process in chunks to handle larger documents
    max_chars = 4000  # Smaller chunks for better quality
    chunks = []
    
    for i in range(0, len(raw_text), max_chars):
        chunk = raw_text[i:i + max_chars]
        chunks.append(chunk)
    
    extracted_parts = []
    
    for i, chunk in enumerate(chunks):
        prompt = f"""Extract ONLY the main content from this PDF document segment.
CRITICAL: Remove ALL of the following:
- Headers, footers, page numbers
- Author names, collaboration credits
- Document titles repeated at top of pages
- Table of contents entries
- Any metadata about the document itself

Keep ONLY:
- Actual content about nutrition, diet, health
- Explanations, recommendations, guidelines
- Lists of foods, recipes, instructions

Document: {filename} (Part {i+1}/{len(chunks)})

Raw text:
{chunk}

Output ONLY the cleaned content below:"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a document cleaner. Remove all headers, footers, author names, page numbers, and metadata. Keep only the actual content about the subject matter."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )

            extracted = response.choices[0].message.content
            if extracted and len(extracted) > 50:
                extracted_parts.append(extracted)
            else:
                # If LLM returns empty/short, skip this chunk
                continue
        except Exception as e:
            # Skip chunk if LLM fails - better to have less content than bad content
            print(f"LLM extraction failed for chunk {i+1}: {e}")
            continue

    # Combine all extracted parts
    full_extracted = "\n\n".join(extracted_parts)
    
    # If extraction produced no content, return cleaned raw text as fallback
    if not full_extracted or len(full_extracted) < 100:
        print("LLM extraction produced insufficient content, using fallback")
        return _clean_pdf_text(raw_text, None)
    
    return full_extracted
