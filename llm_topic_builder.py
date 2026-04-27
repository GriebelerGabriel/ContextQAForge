"""LLM-driven topic tree builder (Pass 2).

Takes clean content segments from Pass 1, sends them to the LLM
to organize into a hierarchical topic tree, and converts the result
into a SectionNode tree compatible with the rest of the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from config import PipelineConfig
from models import ContentSegment, Document, SectionNode

logger = logging.getLogger(__name__)

TOPIC_SYSTEM_PROMPT = """\
You are a document organization expert. You will receive a list of content \
segments from a single document, each with an ID, title, and a brief preview. \
Your task is to organize these segments into a hierarchical topic tree.

Rules:
1. Group related segments under meaningful topic names based on what the \
content actually discusses.
2. Create subtopics when segments within a topic cover distinct sub-areas.
3. The tree should be 2-3 levels deep (topic -> subtopic -> detail).
4. A segment may appear under multiple topics if it covers multiple subjects.
5. Topic names should be descriptive and based on content meaning, not on \
original section titles.
6. Every body segment must appear in at least one topic.
7. Order topics logically (background first, then evidence, then \
recommendations, then implementation).

Output ONLY valid JSON matching this schema:
{
  "document_title": "concise document title",
  "topics": [
    {
      "topic": "Topic Name",
      "segment_ids": ["seg_001", "seg_002"],
      "subtopics": [
        {
          "topic": "Subtopic Name",
          "segment_ids": ["seg_001"],
          "subtopics": []
        }
      ]
    }
  ]
}
"""

TOPIC_USER_TEMPLATE = """\
CONTENT SEGMENTS from document: {doc_source}
---
{segment_list}
---

Organize these segments into a hierarchical topic tree and output valid JSON."""


def build_topic_tree(
    segments: List[ContentSegment],
    doc: Document,
    config: PipelineConfig,
    client: Optional[OpenAI] = None,
) -> Optional[SectionNode]:
    """Build a topic tree from content segments using LLM.

    Args:
        segments: List of body content segments from Pass 1.
        doc: Source document.
        config: Pipeline configuration.
        client: Optional OpenAI client.

    Returns:
        Root SectionNode of the topic tree, or None if building fails.
    """
    if not segments:
        logger.warning(f"No segments to build topic tree for {doc.source}")
        return None

    # Check cache
    cache_dir = Path(config.slicer_cache_dir)
    doc_stem = Path(doc.source).stem
    cache_path = cache_dir / f"{doc_stem}.topics.json"

    if cache_path.exists():
        logger.info(f"Loading cached topic tree for {doc.source}")
        return _load_cached_tree(cache_path, doc.source)

    if client is None:
        client = OpenAI(api_key=config.openai_api_key)

    # Format segment list for the prompt
    segment_list = _format_segment_list(segments)
    user_prompt = TOPIC_USER_TEMPLATE.format(
        doc_source=doc.source,
        segment_list=segment_list,
    )

    logger.info(
        f"Building topic tree for {doc.source} "
        f"({len(segments)} segments, ~{len(segment_list)} chars in prompt)"
    )

    # Call LLM with retries
    tree_data = None
    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=config.topic_model,
                messages=[
                    {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            tree_data = json.loads(raw)
            break
        except Exception as e:
            logger.warning(
                f"  Topic builder attempt {attempt + 1}/{config.max_retries} failed: {e}"
            )
            if attempt == config.max_retries - 1:
                logger.warning("  Falling back to flat topic structure")

    if tree_data is None:
        # Fallback: create a flat structure with one node per segment
        return _build_flat_fallback(segments, doc.source)

    # Convert LLM response to SectionNode tree
    root = _convert_to_section_tree(tree_data, segments, doc.source)

    # Cache results
    if root:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _save_cached_tree(cache_path, root)
        logger.info(f"  Cached topic tree to {cache_path}")

    return root


def _format_segment_list(segments: List[ContentSegment]) -> str:
    """Format segments as a compact list for the LLM prompt."""
    lines: List[str] = []
    preview_len = 200
    for seg in segments:
        preview = seg.content[:preview_len]
        if len(seg.content) > preview_len:
            preview += "..."
        lines.append(f"[{seg.id}] {seg.title}")
        lines.append(f"  Preview: {preview}")
        lines.append("")
    return "\n".join(lines)


def _convert_to_section_tree(
    tree_data: dict,
    segments: List[ContentSegment],
    source: str,
) -> Optional[SectionNode]:
    """Convert the LLM's JSON response into a SectionNode tree."""
    # Build a lookup from segment ID to content
    seg_map: dict = {s.id: s for s in segments}

    # Track which segments have been assigned
    assigned_ids: set = set()

    doc_title = tree_data.get("document_title", "Untitled Document")
    topics = tree_data.get("topics", [])

    if not topics:
        return _build_flat_fallback(segments, source)

    # Build child nodes from topics
    children: List[SectionNode] = []
    for topic in topics:
        node = _build_topic_node(topic, seg_map, source, depth=1, assigned_ids=assigned_ids)
        if node:
            children.append(node)

    # Collect any unassigned segments into an "Other" node
    unassigned = [s for s in segments if s.id not in assigned_ids]
    if unassigned:
        other_node = SectionNode(
            name="Other Content",
            source=source,
            depth=1,
            content="\n\n".join(s.content for s in unassigned),
        )
        children.append(other_node)
        logger.info(f"  {len(unassigned)} segments assigned to 'Other Content'")

    if not children:
        return _build_flat_fallback(segments, source)

    return SectionNode(
        name=doc_title,
        source=source,
        depth=0,
        children=children,
    )


def _build_topic_node(
    topic: dict,
    seg_map: dict,
    source: str,
    depth: int,
    assigned_ids: set,
) -> Optional[SectionNode]:
    """Recursively build a SectionNode from a topic entry."""
    topic_name = topic.get("topic", "Untitled Topic")
    segment_ids = topic.get("segment_ids", [])
    subtopics = topic.get("subtopics", [])

    # Resolve segment IDs to content
    contents: List[str] = []
    for sid in segment_ids:
        if sid in seg_map:
            contents.append(seg_map[sid].content)
            assigned_ids.add(sid)

    # Build children from subtopics
    children: List[SectionNode] = []
    for sub in subtopics:
        child = _build_topic_node(sub, seg_map, source, depth + 1, assigned_ids)
        if child:
            children.append(child)

    # Content: concatenate all segment text for this topic
    content = "\n\n".join(contents) if contents else None

    # If this node has no content and no children, skip it
    if not content and not children:
        return None

    return SectionNode(
        name=topic_name,
        source=source,
        depth=depth,
        content=content if not children else None,  # non-leaf nodes don't hold content
        children=children,
    )


def _build_flat_fallback(
    segments: List[ContentSegment],
    source: str,
) -> SectionNode:
    """Create a flat tree where each segment is its own leaf node."""
    children: List[SectionNode] = []
    for seg in segments:
        children.append(
            SectionNode(
                name=seg.title,
                source=source,
                depth=1,
                content=seg.content,
            )
        )
    return SectionNode(
        name=Path(source).stem,
        source=source,
        depth=0,
        children=children,
    )


def _load_cached_tree(cache_path: Path, source: str) -> Optional[SectionNode]:
    """Load a cached SectionNode tree from JSON."""
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SectionNode.model_validate(data)


def _save_cached_tree(cache_path: Path, root: SectionNode) -> None:
    """Save a SectionNode tree to JSON."""
    data = root.model_dump()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
