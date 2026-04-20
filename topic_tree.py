"""Document-grounded topic tree module.

Builds a hierarchical topic tree from actual document structure (headings),
then optionally refines it with LLM. Each leaf node holds section content
that gets chunked for QA generation.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from config import PipelineConfig
from models import Document, SectionNode

logger = logging.getLogger(__name__)


class DocumentTopicTree:
    """Hierarchical topic tree extracted from document headings.

    Pipeline:
    1. Parse heading hierarchy from structured Markdown → skeleton tree
    2. (Optional) LLM refines sections that are too long or vague
    3. Chunk leaf node content for QA generation
    4. Distribute QA generation across all leaf nodes (round-robin)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.root: Optional[SectionNode] = None

    # ------------------------------------------------------------------ #
    #  Step 1: Build skeleton from document headings                      #
    # ------------------------------------------------------------------ #

    def build_from_documents(self, documents: List[Document]) -> None:
        """Build tree from document heading structure.

        Each document contributes its section hierarchy as a subtree
        under the root node.
        """
        children: List[SectionNode] = []

        for doc in documents:
            doc_tree = self._parse_document_headings(doc)
            if doc_tree:
                children.append(doc_tree)

        if not children:
            # Fallback: create a single node per document
            for doc in documents:
                children.append(SectionNode(
                    name=doc.metadata.get("filename", doc.source),
                    source=doc.source,
                    depth=0,
                    content=doc.content,
                ))

        self.root = SectionNode(
            name="Document Topics",
            source="root",
            depth=0,
            children=children,
        )

        # Phase 2b: LLM refinement
        self._refine_with_llm(self.root)

    def _parse_document_headings(self, doc: Document) -> Optional[SectionNode]:
        """Parse heading hierarchy from a single document's Markdown content."""
        lines = doc.content.split("\n")
        source = doc.source

        # Build a flat list of (level, title) from heading lines
        headings: List[Tuple[int, str, int]] = []  # (level, title, line_index)
        for i, line in enumerate(lines):
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append((level, title, i))

        # If no headings found, return the whole document as a single node
        if not headings:
            return SectionNode(
                name=Path(source).stem,
                source=source,
                depth=0,
                content=doc.content.strip(),
            )

        # Use the first heading as the document root
        root_level = headings[0][0]
        root_title = headings[0][1]

        # Build tree structure using a stack-based approach
        root_node = SectionNode(
            name=root_title,
            source=source,
            depth=0,
        )

        # Stack of (node, heading_level) for nesting
        stack: List[Tuple[SectionNode, int]] = [(root_node, root_level)]

        for idx in range(len(headings)):
            level, title, line_idx = headings[idx]

            # Skip the root heading (already added)
            if idx == 0:
                # Assign content between root heading and next heading
                next_line = headings[1][2] if len(headings) > 1 else len(lines)
                content = "\n".join(lines[line_idx + 1:next_line]).strip()
                if content:
                    root_node.content = content
                continue

            # Find content: text between this heading and the next
            next_line = headings[idx + 1][2] if idx + 1 < len(headings) else len(lines)
            content = "\n".join(lines[line_idx + 1:next_line]).strip()

            new_node = SectionNode(
                name=title,
                source=source,
                depth=0,  # Will be set by parent
                content=content if content else None,
            )

            # Find the correct parent: pop stack until we find a lower level
            while len(stack) > 1 and stack[-1][1] >= level:
                stack.pop()

            parent_node = stack[-1][0]
            new_node.depth = parent_node.depth + 1
            parent_node.children.append(new_node)
            stack.append((new_node, level))

        # If root has no content and no children got the content,
        # the root is just a container — flatten if single child
        if not root_node.content and len(root_node.children) == 0:
            return None

        return root_node

    # ------------------------------------------------------------------ #
    #  Step 2b: LLM refinement                                           #
    # ------------------------------------------------------------------ #

    def _refine_with_llm(self, node: SectionNode) -> None:
        """Recursively refine the tree with LLM.

        Splits long leaf sections into sub-topics, improves vague headings.
        Uses the cheaper topic_model (gpt-4o-mini).
        """
        if node.is_leaf and node.content:
            # Check if section is long enough to warrant splitting
            if len(node.content) > self.config.max_section_chars_for_split:
                self._llm_split_section(node)

        for child in node.children:
            self._refine_with_llm(child)

    def _llm_split_section(self, node: SectionNode) -> None:
        """Use LLM to split a long section into sub-topics."""
        content_preview = node.content[:3000]  # Limit to avoid token overflow

        prompt = f"""Analyze the following text section titled "{node.name}" and split it into 2-5 distinct sub-topics.

For each sub-topic, provide:
1. A clear, descriptive heading
2. The relevant excerpt from the text

Section content:
{content_preview}

Output a JSON array where each element has "name" (sub-topic heading) and "content" (relevant text excerpt).
Output ONLY the JSON array, no other text:
[
  {{"name": "...", "content": "..."}},
  ...
]"""

        for attempt in range(2):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 8))
            try:
                response = self.client.chat.completions.create(
                    model=self.config.topic_model,
                    messages=[
                        {"role": "system", "content": "You are a document structure analyst. Split text into clear sub-topics. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )

                raw = response.choices[0].message.content.strip()
                # Strip markdown code blocks
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]

                sub_topics = json.loads(raw.strip())
                if not isinstance(sub_topics, list) or len(sub_topics) == 0:
                    continue

                # Create child nodes from sub-topics
                for st in sub_topics[:5]:
                    if isinstance(st, dict) and "name" in st and "content" in st:
                        child = SectionNode(
                            name=str(st["name"]),
                            source=node.source,
                            depth=node.depth + 1,
                            content=str(st["content"]),
                        )
                        node.children.append(child)

                # Clear parent content — it's now distributed to children
                node.content = None
                logger.info(f"LLM split '{node.name}' into {len(node.children)} sub-topics")
                return

            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"LLM split failed for '{node.name}': {e}")
                continue

        # If LLM splitting failed, leave node as-is

    # ------------------------------------------------------------------ #
    #  Step 3: Chunk leaf nodes                                          #
    # ------------------------------------------------------------------ #

    def chunk_leaves(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Split each leaf node's content into smaller chunks."""
        if not self.root:
            return

        self._chunk_node(self.root, chunk_size, chunk_overlap)

    def _chunk_node(self, node: SectionNode, chunk_size: int, chunk_overlap: int) -> None:
        """Recursively chunk leaf nodes."""
        if node.is_leaf and node.content:
            node.chunks = self._split_text(node.content, chunk_size, chunk_overlap)

        for child in node.children:
            self._chunk_node(child, chunk_size, chunk_overlap)

    @staticmethod
    def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Find last space for clean boundary
            if end < text_len:
                search_start = max(start, end - 100)
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, end - chunk_overlap)

        return chunks

    # ------------------------------------------------------------------ #
    #  Step 4: QA distribution across leaf nodes                         #
    # ------------------------------------------------------------------ #

    def get_all_leaf_nodes(self) -> List[SectionNode]:
        """Get all leaf nodes in the tree (depth-first, maintains document order)."""
        if not self.root:
            return []
        leaves: List[SectionNode] = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: SectionNode, leaves: List[SectionNode]) -> None:
        """Recursively collect leaf nodes."""
        if node.is_leaf:
            leaves.append(node)
        for child in node.children:
            self._collect_leaves(child, leaves)

    def get_qa_distribution(self, num_samples: int) -> List[Tuple[SectionNode, int]]:
        """Distribute QA count across leaf nodes using round-robin.

        Returns list of (leaf_node, num_qa_pairs) tuples.
        Every leaf gets at least floor(num_samples / num_leaves) pairs.
        Remainder is distributed round-robin starting from leaf 0.
        """
        leaves = self.get_all_leaf_nodes()
        if not leaves:
            return []

        n = len(leaves)
        base_count = num_samples // n
        remainder = num_samples % n

        distribution = []
        for i, leaf in enumerate(leaves):
            count = base_count + (1 if i < remainder else 0)
            if count > 0:
                distribution.append((leaf, count))

        return distribution

    def get_leaf_path(self, node: SectionNode) -> List[str]:
        """Get the full path from root to a node."""
        if not self.root:
            return [node.name]
        path = self._find_path(self.root, node)
        return path if path else [node.name]

    def _find_path(self, current: SectionNode, target: SectionNode) -> Optional[List[str]]:
        """Find path from current node to target."""
        if current is target:
            return [current.name]
        for child in current.children:
            path = self._find_path(child, target)
            if path is not None:
                return [current.name] + path
        return None

    # ------------------------------------------------------------------ #
    #  Visualization                                                      #
    # ------------------------------------------------------------------ #

    def visualize(self) -> str:
        """Return ASCII tree visualization."""
        if not self.root:
            return "No tree built yet."

        lines = ["\n" + "=" * 50, "TOPIC TREE STRUCTURE", "=" * 50]
        self._visualize_node(self.root, "", True, lines)

        leaves = self.get_all_leaf_nodes()
        total_chunks = sum(len(l.chunks) for l in leaves)
        lines.append("=" * 50)
        lines.append(f"Leaf nodes: {len(leaves)}")
        lines.append(f"Total chunks: {total_chunks}")
        lines.append("")
        return "\n".join(lines)

    def _visualize_node(self, node: SectionNode, prefix: str, is_last: bool, lines: List[str]) -> None:
        """Recursively build tree visualization."""
        connector = "└── " if is_last else "├── "
        chunk_info = f" [{len(node.chunks)} chunks]" if node.chunks else ""
        content_info = f" ({len(node.content)} chars)" if node.content else ""
        lines.append(f"{prefix}{connector}{node.name}{content_info}{chunk_info}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            self._visualize_node(child, child_prefix, i == len(node.children) - 1, lines)

    # ------------------------------------------------------------------ #
    #  Save / Load                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save tree to JSON file."""
        if not self.root:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = self.root.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Tree saved to {path}")

    def load(self, path: str) -> None:
        """Load tree from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.root = SectionNode.model_validate(data)
        logger.info(f"Tree loaded from {path} ({len(self.get_all_leaf_nodes())} leaf nodes)")
