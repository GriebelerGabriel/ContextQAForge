"""LLM-driven document content slicer (Pass 1).

Splits document text into manageable chunks with Python first, then sends
each chunk to the LLM to extract substantive content, discarding boilerplate.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from config import PipelineConfig
from models import ContentSegment, Document

logger = logging.getLogger(__name__)

CHUNK_CHAR_LIMIT = 12000
CHUNK_OVERLAP_CHARS = 500
MIN_SEGMENT_CHARS = 200
MAX_SEGMENTS_PER_CHUNK = 6
DEDUP_OVERLAP_THRESHOLD = 0.7

SLICER_SYSTEM_PROMPT = """\
You are a document content analyst. You will receive a PORTION of a document \
extracted from a PDF. Your task is to extract ONLY the substantive content \
from this portion, organized into content blocks.

SUBSTANTIVE CONTENT (keep these):
- Factual explanations, evidence summaries, research findings
- Recommendations, guidelines, actionable advice
- Detailed descriptions of methods, results, or data
- Practical instructions, recipes, meal plans
- Background information that provides necessary context for recommendations
- Tables with meaningful data (recommendations, comparisons, outcomes)

DISCARD (do not include these):
- Copyright notices, publisher information, ISBN numbers
- Table of contents listings
- Author lists, committee member lists, stakeholder panels
- Acknowledgements, disclosures, conflict of interest statements
- Repeating headers/footers (page numbers, document titles on every page)
- References/bibliography sections (numbered citation lists)
- Abbreviation/acronym lists
- Administrative/process descriptions about how a guideline was developed
- Empty or near-empty sections

Output ONLY valid JSON matching this schema:
{
  "segments": [
    {
      "id": "seg_001",
      "title": "descriptive title for the section",
      "content": "the verbatim text content"
    }
  ]
}

CRITICAL RULES:
1. HARD LIMIT: Output AT MOST 6 segments. If the text has fewer topics, output fewer segments.
2. Each segment MUST be at least 300 characters. If content is shorter, merge it with an adjacent topic.
3. COPY the text VERBATIM from the input. Do NOT paraphrase, summarize, rewrite, or invent content.
4. Each segment MUST cover a DIFFERENT topic or section. Do NOT repeat the same content with different titles.
5. Segment IDs must be sequential: seg_001, seg_002, seg_003, etc.
6. If this portion contains only boilerplate (headers, footers, copyright), output an empty segments array: {"segments": []}
"""

SLICER_USER_TEMPLATE = """\
DOCUMENT TEXT (portion {chunk_index} of {total_chunks}):
---
{text}
---

Extract the substantive content from this portion, discarding all boilerplate. \
Output valid JSON with only the relevant segments."""


def _split_text_into_chunks(text: str, limit: int = CHUNK_CHAR_LIMIT, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + limit, text_len)
        if end < text_len:
            # Try to break at a paragraph boundary
            para_break = text.rfind("\n\n", start + limit // 2, end)
            if para_break > start:
                end = para_break
            else:
                # Fall back to last space
                last_space = text.rfind(" ", end - 200, end)
                if last_space > start:
                    end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end - overlap)

    return chunks


def slice_document(
    doc: Document,
    config: PipelineConfig,
    client: Optional[OpenAI] = None,
) -> List[ContentSegment]:
    """Slice a document into clean content segments using LLM classification.

    Splits the text into manageable chunks with Python first, then sends
    each chunk to the LLM separately to avoid output token limits.

    Args:
        doc: Document to slice.
        config: Pipeline configuration.
        client: Optional OpenAI client (created from config if not provided).

    Returns:
        List of ContentSegment (body content only).
    """
    # Check cache
    cache_dir = Path(config.slicer_cache_dir)
    doc_stem = Path(doc.source).stem
    cache_path = cache_dir / f"{doc_stem}.sliced.json"

    if cache_path.exists():
        cached = _load_cached_segments(cache_path)
        if cached:
            logger.info(f"Loading cached slicing results for {doc.source}")
            return cached
        logger.info(f"Cache empty for {doc.source}, re-processing")

    if client is None:
        client = OpenAI(api_key=config.openai_api_key)

    text_chunks = _split_text_into_chunks(doc.content)
    logger.info(
        f"Slicing {doc.source} ({len(doc.content)} chars, "
        f"split into {len(text_chunks)} chunks)"
    )

    all_segments: List[ContentSegment] = []
    seg_counter = 0

    for i, chunk in enumerate(text_chunks):
        segments = _call_llm(
            chunk,
            chunk_index=i + 1,
            total_chunks=len(text_chunks),
            source=doc.source,
            client=client,
            model=config.slicer_model,
            max_retries=config.max_retries,
        )
        # Re-number segment IDs globally
        for seg in segments:
            seg_counter += 1
            seg.id = f"seg_{seg_counter:03d}"
            all_segments.append(seg)

    # Global deduplication across chunks
    all_segments = _deduplicate_segments(all_segments)

    logger.info(f"  Got {len(all_segments)} content segments from LLM")

    if not all_segments:
        logger.warning("  No segments produced, falling back to entire document as one segment")
        all_segments = [
            ContentSegment(
                id="seg_001",
                title="Full document (fallback)",
                content=doc.content,
                content_type="body",
                source=doc.source,
            )
        ]

    # Cache results
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_cached_segments(cache_path, all_segments)
    logger.info(f"  Cached slicing results to {cache_path}")

    return all_segments


def _call_llm(
    text: str,
    chunk_index: int,
    total_chunks: int,
    source: str,
    client: OpenAI,
    model: str,
    max_retries: int = 3,
) -> List[ContentSegment]:
    """Send a text chunk to the LLM and parse the response."""
    user_prompt = SLICER_USER_TEMPLATE.format(
        text=text, chunk_index=chunk_index, total_chunks=total_chunks,
    )

    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(min(2 ** attempt, 16))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SLICER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from LLM")
            raw = response.choices[0].message.content
            return _parse_response(raw, source)
        except Exception as e:
            logger.warning(f"  Slicer chunk {chunk_index} attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                # Fallback: return this chunk as a single segment
                logger.warning(f"  Falling back to returning chunk {chunk_index} as one segment")
                return [
                    ContentSegment(
                        id="seg_001",
                        title=f"Chunk {chunk_index} (fallback)",
                        content=text,
                        content_type="body",
                        source=source,
                    )
                ]

    return []


def _parse_response(raw: str, source: str) -> List[ContentSegment]:
    """Parse the LLM JSON response into ContentSegment objects.

    Enforces minimum content length and caps segment count per chunk.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("  Failed to parse slicer response as JSON")
        return []

    raw_segments = data.get("segments", [])
    if not raw_segments:
        return []

    segments: List[ContentSegment] = []
    for i, seg_data in enumerate(raw_segments):
        if i >= MAX_SEGMENTS_PER_CHUNK:
            logger.info(f"  Capping at {MAX_SEGMENTS_PER_CHUNK} segments (LLM returned {len(raw_segments)})")
            break
        try:
            content = str(seg_data.get("content", "")).strip()
            if len(content) < MIN_SEGMENT_CHARS:
                continue
            seg = ContentSegment(
                id=f"seg_{i + 1:03d}",
                title=str(seg_data.get("title", f"Segment {i + 1}")),
                content=content,
                content_type="body",
                source=source,
            )
            segments.append(seg)
        except Exception as e:
            logger.warning(f"  Failed to parse segment {i}: {e}")
            continue

    return segments


def _deduplicate_segments(segments: List[ContentSegment]) -> List[ContentSegment]:
    """Remove segments with highly overlapping content."""
    if not segments:
        return segments

    unique: List[ContentSegment] = []
    for seg in segments:
        seg_words = set(seg.content.lower().split())
        is_dup = False
        for existing in unique:
            existing_words = set(existing.content.lower().split())
            if not seg_words or not existing_words:
                continue
            overlap = len(seg_words & existing_words) / max(len(seg_words), len(existing_words))
            if overlap > DEDUP_OVERLAP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            unique.append(seg)
        else:
            logger.info(f"  Deduplicating segment: {seg.title[:50]}...")

    if len(unique) < len(segments):
        logger.info(f"  Removed {len(segments) - len(unique)} duplicate segments")

    return unique


def _load_cached_segments(cache_path: Path) -> List[ContentSegment]:
    """Load cached segments from a JSON file."""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ContentSegment.model_validate(s) for s in data]
    except Exception as e:
        logger.warning(f"  Corrupted cache file {cache_path}: {e}. Re-processing.")
        cache_path.unlink(missing_ok=True)
        return []


def _save_cached_segments(cache_path: Path, segments: List[ContentSegment]) -> None:
    """Save segments to a JSON cache file."""
    data = [s.model_dump() for s in segments]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
