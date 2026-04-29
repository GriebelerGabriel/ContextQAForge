"""Document-grounded topic tree module.

Builds a hierarchical topic tree from actual document structure (headings).
Each leaf node holds section content that gets chunked for QA generation.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from config import PipelineConfig
from models import Document, SectionNode

logger = logging.getLogger(__name__)


class DocumentTopicTree:
    """Hierarchical topic tree extracted from document headings.

    Pipeline:
    1. Parse heading hierarchy from structured Markdown → skeleton tree
    2. Filter out non-content sections (copyright, TOC, person lists, etc.)
    3. Chunk leaf node content for QA generation
    4. Distribute QA generation across all leaf nodes (round-robin)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.root: Optional[SectionNode] = None

    # ------------------------------------------------------------------ #
    #  Step 1: Build skeleton from document headings                      #
    # ------------------------------------------------------------------ #

    def build_from_documents(self, documents: List[Document]) -> None:
        """Build tree from documents.

        Uses LLM-driven pipeline (slicing + topic building) when
        config.use_llm_pipeline is True, otherwise falls back to
        heading-based parsing.
        """
        if getattr(self.config, "use_llm_pipeline", False):
            self._build_llm(documents)
        else:
            self._build_heading_based(documents)

    # ------------------------------------------------------------------ #
    #  LLM-driven pipeline                                                #
    # ------------------------------------------------------------------ #

    def _build_llm(self, documents: List[Document]) -> None:
        """Build tree using LLM content slicing and topic organization."""
        from llm_slicer import slice_document
        from llm_topic_builder import build_topic_tree

        children: List[SectionNode] = []
        num_docs = max(len(documents), 1)
        target_per_doc = max(self.config.num_samples // num_docs, 1)

        for doc in documents:
            logger.info(f"LLM pipeline: processing {doc.source}")
            segments = slice_document(doc, self.config)
            if not segments:
                logger.warning(f"No segments produced for {doc.source}, skipping")
                continue

            # Cap leaves: at most 2x the number of segments — keeps topics
            # meaningful and leaves with enough content for good QA grounding
            target_leaves = min(target_per_doc, len(segments) * 2)
            tree = build_topic_tree(segments, doc, self.config, target_leaves=target_leaves)
            if tree:
                filename = doc.metadata.get("filename", doc.source)
                file_node = SectionNode(
                    name=filename,
                    source=doc.source,
                    depth=0,
                    children=[tree],
                )
                tree.depth = 1
                children.append(file_node)

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

    # ------------------------------------------------------------------ #
    #  Heading-based pipeline (original)                                  #
    # ------------------------------------------------------------------ #

    def _build_heading_based(self, documents: List[Document]) -> None:
        """Build tree from document heading structure (original pipeline).

        Each document is wrapped in a file-level node (not used for QA).
        Under it, the document's heading hierarchy forms the content tree.
        """
        children: List[SectionNode] = []

        for doc in documents:
            doc_tree = self._parse_document_headings(doc)
            if doc_tree:
                doc_tree = self._filter_non_content(doc_tree)
            if doc_tree:
                # Wrap in a file-level node so the tree shows which file
                # each section comes from. File nodes have no content and
                # always have children → never leaf → never used for QA.
                filename = doc.metadata.get("filename", doc.source)
                file_node = SectionNode(
                    name=filename,
                    source=doc.source,
                    depth=0,
                    children=[doc_tree],
                )
                doc_tree.depth = 1
                children.append(file_node)

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

    # ------------------------------------------------------------------ #
    #  Content section filtering                                          #
    # ------------------------------------------------------------------ #

    # Patterns for sections that are NOT real document content
    _SKIP_SECTION_PATTERNS = [
        r"©",
        r"all rights reserved",
        r"copyright",
        r"table of contents",
        r"^contents\s*$",
        r"abbreviations?\s+(and|&)\s+acronyms?",
        r"^abbreviations?$",
        r"^acknowledgements?$",
        r"members?\s+of\s+(the|who)",
        r"external\s+experts?",
        r"stakeholder\s+panel",
        r"^disclosures?$",
        r"conflict\s+of\s+interest",
        r"management\s+of\s+confl",
        r"^references?$",
        r"article\s+information",
        r"permissions\s+and\s+copyright",
        r"^annex\s+\d+",
        # Portuguese patterns
        r"^sumário?$",
        r"^agradecimentos?$",
        r"^referências?$",
        r"^anexo\s+\d+",
        r"^siglas?\s+e\s+abreviaturas?",
    ]
    _SKIP_COMPILED = re.compile(
        "|".join(_SKIP_SECTION_PATTERNS), re.IGNORECASE
    )

    @classmethod
    def _is_content_section(cls, name: str, content: Optional[str]) -> bool:
        """Check if a section contains real document content (not front matter)."""
        if cls._SKIP_COMPILED.search(name):
            return False
        # Skip leaf nodes with too little content
        if content is not None and len(content) < 50:
            return False
        return True

    # Patterns indicating front-matter text (copyright, ISBN, publisher info, TOC listings)
    _FRONT_MATTER_PATTERNS = [
        r"©",
        r"all rights reserved",
        r"copyright",
        r"ISBN",
        r"WHO Press",
        r"World Health Organization.*20 Avenue Appia",
        r"Requests for permission to reproduce",
        r"designations employed",
        r"printed by",
        r"suggested citation",
        r"WHO Library Cataloguing",
        r"^\d+\.\s+\w+.*\d+$",  # TOC line like "Acknowledgements  VII"
    ]
    _FRONT_MATTER_COMPILED = re.compile(
        "|".join(_FRONT_MATTER_PATTERNS), re.IGNORECASE
    )

    @classmethod
    def _is_front_matter(cls, text: str) -> bool:
        """Check if text is predominantly front matter (copyright, TOC, etc.)."""
        if not text:
            return False
        lines = text.split("\n")
        front_matter_lines = sum(
            1 for line in lines if cls._FRONT_MATTER_COMPILED.search(line)
        )
        # If more than 30% of non-empty lines are front matter, consider it all front matter
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return True
        return front_matter_lines / len(non_empty) > 0.3

    def _filter_non_content(self, node: SectionNode) -> Optional[SectionNode]:
        """Recursively remove non-content sections from the tree."""
        # Skip entire subtrees whose name matches skip patterns
        if self._SKIP_COMPILED.search(node.name):
            return None

        # Filter children recursively
        filtered_children = []
        for child in node.children:
            filtered = self._filter_non_content(child)
            if filtered:
                filtered_children.append(filtered)
        node.children = filtered_children

        # If this is a leaf (no children left), check if it's content
        if not node.children:
            # Leaf with no content — skip
            if not node.content:
                return None
            # Skip leaf nodes with too little content
            if len(node.content) < 50:
                return None

        # For non-leaf nodes, strip content that is purely front matter
        # (copyright, ISBN, TOC listings, etc.) since real content is in children
        if node.children and node.content and self._is_front_matter(node.content):
            node.content = None

        return node

    # ------------------------------------------------------------------ #
    #  Step 1: Build skeleton from document headings                      #
    # ------------------------------------------------------------------ #

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

            # Find last space for clean boundary (only if not at the end)
            if end < text_len:
                search_start = max(start, end - 100)
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # If we've reached the end of text, stop
            if end >= text_len:
                break

            start = max(start + 1, end - chunk_overlap)

        return chunks

    # ------------------------------------------------------------------ #
    #  Step 4: QA distribution across leaf nodes (round-robin)           #
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

    def _is_file_node(self, node: SectionNode) -> bool:
        """Check if a node is a file-level wrapper (named after source file)."""
        if node.source == "root":
            return False
        name_lower = node.name.lower()
        return name_lower.endswith((".pdf", ".txt", ".md"))

    def _visualize_node(self, node: SectionNode, prefix: str, is_last: bool, lines: List[str]) -> None:
        """Recursively build tree visualization."""
        connector = "└── " if is_last else "├── "
        chunk_info = f" [{len(node.chunks)} chunks]" if node.chunks else ""
        content_info = f" ({len(node.content)} chars)" if node.content else ""
        file_marker = "[FILE] " if self._is_file_node(node) else ""
        lines.append(f"{prefix}{connector}{file_marker}{node.name}{content_info}{chunk_info}")

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
