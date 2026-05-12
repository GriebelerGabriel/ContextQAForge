"""LLM-driven topic tree builder (Pass 2).

Takes clean content segments from Pass 1, sends them to the LLM
to organize into a hierarchical topic tree, and converts the result
into a SectionNode tree compatible with the rest of the pipeline.
"""

import json
import logging
import time
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
8. IMPORTANT: The target number of LEAF nodes (nodes with no subtopics) is \
{target_leaves}. Split or merge topics to get close to this number. \
More leaves means more diverse QA coverage.

Output ONLY valid JSON matching this schema:
{{
  "document_title": "concise document title",
  "topics": [
    {{
      "topic": "Topic Name",
      "segment_ids": ["seg_001", "seg_002"],
      "subtopics": [
        {{
          "topic": "Subtopic Name",
          "segment_ids": ["seg_001"],
          "subtopics": []
        }}
      ]
    }}
  ]
}}
"""

TOPIC_USER_TEMPLATE = """\
CONTENT SEGMENTS from document: {doc_source}
TARGET LEAF NODES for this document: {target_leaves} \
(give or take 1-2, but aim for this number)
---
{segment_list}
---

Organize these segments into a hierarchical topic tree with approximately \
{target_leaves} leaf nodes and output valid JSON."""


def build_topic_tree(
    segments: List[ContentSegment],
    doc: Document,
    config: PipelineConfig,
    client: Optional[OpenAI] = None,
    target_leaves: int = 0,
) -> Optional[SectionNode]:
    """Build a topic tree from content segments using LLM.

    Args:
        segments: List of body content segments from Pass 1.
        doc: Source document.
        config: Pipeline configuration.
        client: Optional OpenAI client.
        target_leaves: Approximate number of leaf nodes to create.

    Returns:
        Root SectionNode of the topic tree, or None if building fails.
    """
    if not segments:
        logger.warning(f"No segments to build topic tree for {doc.source}")
        return None

    # Default: aim for ~1 leaf per segment if not specified
    if target_leaves <= 0:
        target_leaves = len(segments)

    # Check cache
    cache_dir = Path(config.slicer_cache_dir)
    doc_stem = Path(doc.source).stem
    cache_path = cache_dir / f"{doc_stem}.topics.json"

    if cache_path.exists():
        cached = _load_cached_tree(cache_path, doc.source)
        if cached:
            logger.info(f"Loading cached topic tree for {doc.source}")
            return cached
        logger.info(f"Topic cache empty for {doc.source}, re-processing")

    if client is None:
        client = OpenAI(api_key=config.openai_api_key)

    # Format segment list for the prompt
    segment_list = _format_segment_list(segments)
    system_prompt = TOPIC_SYSTEM_PROMPT.format(target_leaves=target_leaves)
    user_prompt = TOPIC_USER_TEMPLATE.format(
        doc_source=doc.source,
        target_leaves=target_leaves,
        segment_list=segment_list,
    )

    logger.info(
        f"Building topic tree for {doc.source} "
        f"({len(segments)} segments, ~{len(segment_list)} chars in prompt)"
    )

    # Call LLM with retries
    tree_data = None
    for attempt in range(config.max_retries):
        if attempt > 0:
            time.sleep(min(2 ** attempt, 16))
        try:
            response = client.chat.completions.create(
                model=config.topic_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from LLM")
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
    preview_len = 500
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

    # Collect any unassigned segments and try a second pass to assign them
    unassigned = [s for s in segments if s.id not in assigned_ids]
    if unassigned:
        logger.info(f"  {len(unassigned)} segments unassigned, attempting reclassification...")
        reassigned = _reclassify_unassigned(
            unassigned, children, doc_title, source, assigned_ids,
        )
        if reassigned:
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

    # If this node has both content and children, preserve the content
    # by creating an "Overview" child so it isn't discarded.
    if content and children:
        overview_node = SectionNode(
            name="Overview",
            source=source,
            depth=depth + 1,
            content=content,
        )
        children = [overview_node] + children

    return SectionNode(
        name=topic_name,
        source=source,
        depth=depth,
        content=content if not children else None,
        children=children,
    )


def _reclassify_unassigned(
    unassigned: List[ContentSegment],
    existing_children: List[SectionNode],
    doc_title: str,
    source: str,
    assigned_ids: set,
) -> bool:
    """Try to assign leftover segments into the existing topic tree via a second LLM call.

    Returns True if any segments were successfully reclassified.
    """
    # Collect existing topic names for context
    topic_names = []
    for child in existing_children:
        if child.name == "Other Content":
            continue
        topic_names.append(child.name)
        for sub in child.children:
            topic_names.append(f"  {sub.name}")

    segment_list = "\n".join(
        f"[{s.id}] {s.title}\n  Preview: {s.content[:300]}{'...' if len(s.content) > 300 else ''}\n"
        for s in unassigned
    )

    prompt = f"""The following segments from "{doc_title}" were NOT assigned to any topic in the previous classification.
Existing topics in the tree:
{chr(10).join(topic_names)}

Unassigned segments:
{segment_list}

Assign EACH segment to the most fitting existing topic or subtopic.
If a segment does not fit any existing topic, create a new subtopic under the closest parent.

Output ONLY valid JSON:
{{
  "assignments": [
    {{"segment_id": "seg_XXX", "topic": "Topic Name", "subtopic": "Subtopic Name or null"}},
    ...
  ]
}}"""

    try:
        client = OpenAI(api_key=PipelineConfig.from_env().openai_api_key)
        response = client.chat.completions.create(
            model=PipelineConfig.from_env().topic_model,
            messages=[
                {"role": "system", "content": "You are a document classification assistant. Assign every segment to a topic."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        assignments = data.get("assignments", [])

        # Build a flat map of all leaf nodes by name for quick lookup
        leaf_map: dict = {}
        for child in existing_children:
            if not child.children:
                leaf_map[child.name] = child
            for sub in child.children:
                if not sub.children:
                    leaf_map[sub.name] = sub
                for detail in sub.children:
                    leaf_map[detail.name] = detail

        seg_map = {s.id: s for s in unassigned}
        reassigned_any = False

        for assignment in assignments:
            sid = assignment.get("segment_id", "")
            if sid not in seg_map:
                continue

            # Find target node: prefer subtopic, then topic
            target_name = assignment.get("subtopic") or assignment.get("topic")
            target = leaf_map.get(target_name) if target_name else None

            if target:
                seg = seg_map[sid]
                if target.content:
                    target.content += "\n\n" + seg.content
                else:
                    target.content = seg.content
                assigned_ids.add(sid)
                reassigned_any = True

        if reassigned_any:
            logger.info(f"  Reclassified {sum(1 for s in unassigned if s.id in assigned_ids)} previously unassigned segments")

        return reassigned_any

    except Exception as e:
        logger.warning(f"  Reclassification failed: {e}")
        return False


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
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SectionNode.model_validate(data)
    except Exception as e:
        logger.warning(f"  Corrupted topic cache {cache_path}: {e}. Re-processing.")
        cache_path.unlink(missing_ok=True)
        return None


def _save_cached_tree(cache_path: Path, root: SectionNode) -> None:
    """Save a SectionNode tree to JSON."""
    data = root.model_dump()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
