"""LLM-driven document content slicer (Pass 1).

Sends the entire document text to the LLM and asks it to slice the
document into coherent content blocks, keeping only substantive content
and discarding boilerplate, metadata, TOC, etc.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from config import PipelineConfig
from models import ContentSegment, Document

logger = logging.getLogger(__name__)

SLICER_SYSTEM_PROMPT = """\
You are a document content analyst. You will receive the FULL TEXT of a document \
extracted from a PDF. Your task is to read through the entire document and extract \
ONLY the substantive content, organized into a small number of LARGE, thorough \
content blocks.

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

IMPORTANT RULES:
1. You receive the ENTIRE document. Read it all and produce a single output.
2. Preserve the ORIGINAL TEXT exactly as provided. Do not summarize, paraphrase, \
or omit any substantive content.
3. Create 6-10 segments total per document. Each segment should be at least \
800 characters. Combine related subsections into one segment. A segment of \
1500-3000 chars is ideal. Do NOT create tiny segments under 500 chars.
4. Only include segments with substantive content — skip everything that is boilerplate.
5. Segment IDs must be sequential: seg_001, seg_002, seg_003, etc.
6. When in doubt about whether something is content, INCLUDE it.
"""

SLICER_USER_TEMPLATE = """\
DOCUMENT FULL TEXT:
---
{text}
---

Read through the entire document above. Slice it into substantive content blocks, \
discarding all boilerplate. Output valid JSON with only the relevant segments."""


def slice_document(
    doc: Document,
    config: PipelineConfig,
    client: Optional[OpenAI] = None,
) -> List[ContentSegment]:
    """Slice a document into clean content segments using LLM classification.

    Sends the entire document text to the LLM in one call.

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
        logger.info(f"Loading cached slicing results for {doc.source}")
        return _load_cached_segments(cache_path)

    if client is None:
        client = OpenAI(api_key=config.openai_api_key)

    logger.info(f"Slicing {doc.source} ({len(doc.content)} chars)")

    # Send entire document to LLM
    segments = _call_llm(
        doc.content,
        source=doc.source,
        client=client,
        model=config.slicer_model,
        max_retries=config.max_retries,
    )

    logger.info(
        f"  Got {len(segments)} content segments from LLM"
    )

    # Cache results
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_cached_segments(cache_path, segments)
    logger.info(f"  Cached slicing results to {cache_path}")

    return segments


def _call_llm(
    text: str,
    source: str,
    client: OpenAI,
    model: str,
    max_retries: int = 3,
) -> List[ContentSegment]:
    """Send the full document to the LLM and parse the response."""
    user_prompt = SLICER_USER_TEMPLATE.format(text=text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SLICER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return _parse_response(raw, source)
        except Exception as e:
            logger.warning(f"  Slicer attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                # Fallback: return the entire document as a single segment
                logger.warning("  Falling back to returning entire document as one segment")
                return [
                    ContentSegment(
                        id="seg_001",
                        title="Full document (fallback)",
                        content=text,
                        content_type="body",
                        source=source,
                    )
                ]

    return []


def _parse_response(raw: str, source: str) -> List[ContentSegment]:
    """Parse the LLM JSON response into ContentSegment objects."""
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
        try:
            seg = ContentSegment(
                id=f"seg_{i + 1:03d}",
                title=str(seg_data.get("title", f"Segment {i + 1}")),
                content=str(seg_data.get("content", "")),
                content_type="body",  # LLM already filtered out non-body
                source=source,
            )
            if seg.content:
                segments.append(seg)
        except Exception as e:
            logger.warning(f"  Failed to parse segment {i}: {e}")
            continue

    return segments


def _load_cached_segments(cache_path: Path) -> List[ContentSegment]:
    """Load cached segments from a JSON file."""
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ContentSegment.model_validate(s) for s in data]


def _save_cached_segments(cache_path: Path, segments: List[ContentSegment]) -> None:
    """Save segments to a JSON cache file."""
    data = [s.model_dump() for s in segments]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
