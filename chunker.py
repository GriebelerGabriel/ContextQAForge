"""Text chunking module with configurable size and overlap."""

from typing import List, Optional

from models import Chunk, Document


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    quality_patterns: Optional[List[str]] = None,
) -> List[Chunk]:
    """
    Chunk documents into smaller pieces.

    Args:
        documents: List of documents to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        quality_patterns: Patterns to identify low-quality chunks (headers/footers)

    Returns:
        List of Chunk objects
    """
    chunks = []
    chunk_id = 0

    for doc in documents:
        doc_chunks = _chunk_single_document(doc, chunk_size, chunk_overlap, chunk_id)
        # Filter out low-quality chunks (headers, footers, metadata)
        doc_chunks = [c for c in doc_chunks if _is_quality_content(c.content, quality_patterns)]
        chunks.extend(doc_chunks)
        chunk_id += len(doc_chunks)

    return chunks


def _is_quality_content(content: str, quality_patterns: Optional[List[str]] = None) -> bool:
    """Check if chunk contains quality content (not metadata, boilerplate, or references)."""
    content_lower = content.lower().strip()

    if quality_patterns is None:
        from config import DEFAULT_CHUNK_QUALITY_PATTERNS
        quality_patterns = list(DEFAULT_CHUNK_QUALITY_PATTERNS)

    # Skip if too short
    if len(content) < 100:
        return False

    # Count how many header patterns appear
    pattern_count = 0
    for pattern in quality_patterns:
        if pattern in content_lower:
            pattern_count += 1

    if pattern_count >= 2:
        return False

    # Check if content is mostly header text (high ratio of newlines to content)
    newline_ratio = content.count("\n") / max(len(content), 1)
    if newline_ratio > 0.1:
        return False

    # Check for common header-only words from the quality patterns
    header_words = [p.strip() for p in quality_patterns if len(p.strip().split()) <= 3]
    header_word_count = sum(1 for w in header_words if w in content_lower)
    if header_word_count >= 3:
        return False

    # Filter document metadata/boilerplate
    metadata_indicators = [
        "copyright", "disclosure", "writing group", "writing panel",
        "permission to reproduce", "isbn", "cataloguing-in-publication",
        "endorsed or recommended", "expressed or implied", "warranty",
        "opinion whatsoever", "dotted line", "conflict of interest",
        "who press", "bookorders@", "printed by the who",
        "suggested citation", "annex", "steering committee",
        "declaration of interest", "heart.org/en/about-us",
        "search for guidelines", "browse by topic",
        "meredith.edelman@", "wolterskluwer",
        "writing group disclosures", "ownership interest",
        "research grant support", "honoraria",
        "gratefully acknowledges", "express our deep appreciation",
        "financial support", "abbreviations and acronyms",
        "acknowledgements", "table of contents",
        "low-density lipoprotein", "university of beirut",
        "food and drug administration", "health industry development",
        "nutrition guidance expert advisory",
        "guidelines review committee", "office of the legal counsel",
        "department of nutrition for health and development",
        "grafmac", "cover design and layout",
        "ministry of health, labour and welfare",
        "logistic support", "data collection",
    ]
    metadata_hits = sum(1 for kw in metadata_indicators if kw in content_lower)
    if metadata_hits >= 1 and len(content) < 300:
        return False
    if metadata_hits >= 2:
        return False

    # Filter bibliography/reference entries (lots of short citation fragments)
    citation_patterns = ["et al.", "doi:", "pmid:", "systematic review.", "http://", "https://"]
    citation_hits = sum(1 for p in citation_patterns if p in content_lower)
    if citation_hits >= 2 and len(content) < 300:
        return False

    return True


def _chunk_single_document(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    start_id: int,
) -> List[Chunk]:
    """Chunk a single document into pieces (fast version)."""
    content = doc.content
    chunks = []

    if len(content) <= chunk_size:
        return [
            Chunk(
                content=content,
                source=doc.source,
                chunk_id=start_id,
                start_pos=0,
                end_pos=len(content),
            )
        ]

    start = 0
    chunk_id = start_id
    content_len = len(content)

    while start < content_len:
        end = min(start + chunk_size, content_len)

        # Fast boundary detection: just look for last space in chunk range
        if end < content_len:
            # Find last space within chunk (simpler than sentence detection)
            search_start = max(start, end - 100)  # Only look back 100 chars max
            last_space = content.rfind(" ", search_start, end)
            if last_space > start:
                end = last_space

        chunk_content = content[start:end].strip()
        if chunk_content:
            chunks.append(
                Chunk(
                    content=chunk_content,
                    source=doc.source,
                    chunk_id=chunk_id,
                    start_pos=start,
                    end_pos=end,
                )
            )
            chunk_id += 1

        # Move start position with overlap
        start = max(start + 1, end - chunk_overlap)

    return chunks
