"""Topic tree module for progressive QA specificity like Pluto."""

import json
import random
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from config import PipelineConfig


class TopicNode:
    """A node in the topic tree."""

    def __init__(self, name: str, depth: int = 0):
        self.name = name
        self.depth = depth
        self.children: List["TopicNode"] = []

    def add_child(self, child: "TopicNode") -> None:
        """Add a child node."""
        self.children.append(child)

    def get_path(self) -> List[str]:
        """Get path from root to this node."""
        if self.depth == 0:
            return [self.name]
        return []  # Set by tree builder


class TopicTree:
    """Hierarchical topic tree for progressive question generation."""

    def __init__(
        self,
        config: PipelineConfig,
        root_topic: str = "Document Topics",
        tree_degree: int = 3,  # children per node
        tree_depth: int = 3,  # levels of depth
    ):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.root_topic = root_topic
        self.tree_degree = tree_degree
        self.tree_depth = tree_depth
        self.root: Optional[TopicNode] = None
        self.leaf_paths: List[List[str]] = []  # All root-to-leaf paths
        self.query_seeds: Dict[str, str] = {}  # topic_path_key -> query_seed
        self.current_iteration = 0

    def build_from_chunks(self, chunks: List[str]) -> None:
        """Build topic tree from document chunks."""
        # Build root
        self.root = TopicNode(self.root_topic, depth=0)

        # Generate topics using LLM
        sample_text = "\n\n".join(chunks[:10])  # Sample first 10 chunks
        topics = self._generate_topics(sample_text, self.root.name, depth=0)

        # Build tree recursively
        for topic in topics[: self.tree_degree]:
            child = TopicNode(topic, depth=1)
            self.root.add_child(child)
            self._build_subtree(child, chunks, depth=1)

        # Collect all leaf paths
        self.leaf_paths = self._collect_leaf_paths(self.root)

    def generate_query_seeds(self) -> None:
        """Generate rich query seeds for all leaf paths using a single batch LLM call.

        Each query seed expands the topic into descriptive search terms
        that improve vector retrieval quality.
        """
        if not self.leaf_paths:
            return

        # Build all topics as a single prompt for batch generation
        topics_text = "\n".join([
            f"{i+1}. {' -> '.join(path)}"
            for i, path in enumerate(self.leaf_paths)
        ])

        prompt = f"""Given the following list of topic paths, generate a rich search query for each one.
The search query should expand the topic into specific, descriptive terms that would help retrieve
relevant information from a document database. Include related terms, synonyms, and specifics.

Topics:
{topics_text}

Output a JSON object mapping each topic number (as string key) to its search query.
Example: {{"1": "causes, symptoms, treatment, risk factors of vitamin D deficiency in adults", "2": "..."}}
Output only the JSON object, no other text:"""

        for attempt in range(3):
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))
            try:
                response = self.client.chat.completions.create(
                    model=self.config.topic_model,
                    messages=[
                        {"role": "system", "content": "You are a search query expansion assistant. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                )

                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

                seeds_map = json.loads(content.strip())
                if isinstance(seeds_map, dict):
                    for i, path in enumerate(self.leaf_paths):
                        key = str(i + 1)
                        if key in seeds_map:
                            path_key = " -> ".join(path)
                            self.query_seeds[path_key] = seeds_map[key]
                    return
            except Exception:
                continue

        # Fallback: use topic text as-is
        for path in self.leaf_paths:
            path_key = " -> ".join(path)
            self.query_seeds[path_key] = " ".join(path)

    def get_query_seed(self, topic_path: List[str]) -> str:
        """Get the pre-generated query seed for a topic path.

        Falls back to the full leaf path's seed, or raw topic text if no seed exists.
        """
        # Try exact path match first
        path_key = " -> ".join(topic_path)
        if path_key in self.query_seeds:
            return self.query_seeds[path_key]

        # Try matching against full leaf paths that start with this prefix
        for leaf_path in self.leaf_paths:
            leaf_key = " -> ".join(leaf_path)
            if leaf_key.startswith(path_key) and leaf_key in self.query_seeds:
                return self.query_seeds[leaf_key]

        # Final fallback: join topic words
        return " ".join(topic_path)

    def _build_subtree(self, node: TopicNode, chunks: List[str], depth: int) -> None:
        """Recursively build subtree."""
        if depth >= self.tree_depth:
            return

        # Get path to this node for context
        path = self._get_node_path(self.root, node)
        context = " -> ".join(path)

        # Find relevant chunks using keyword matching
        relevant_chunks = self._find_relevant_chunks(node.name, chunks)[:5]
        if not relevant_chunks:
            relevant_chunks = random.sample(chunks, min(3, len(chunks)))

        sample_text = "\n\n".join(relevant_chunks)

        # Generate subtopics
        subtopics = self._generate_topics(sample_text, context, depth)

        for subtopic in subtopics[: self.tree_degree]:
            child = TopicNode(subtopic, depth=depth + 1)
            node.add_child(child)
            self._build_subtree(child, chunks, depth + 1)

    def _find_relevant_chunks(self, topic_name: str, chunks: List[str], min_keyword_len: int = 4) -> List[str]:
        """Find chunks relevant to a topic using keyword matching."""
        # Split topic into keywords (skip short words)
        keywords = [w.lower() for w in topic_name.split() if len(w) >= min_keyword_len]

        scored = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Count how many keywords appear in the chunk
            score = sum(1 for kw in keywords if kw in chunk_lower)
            if score > 0:
                scored.append((score, chunk))

        # Sort by relevance (most keywords matched first)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored]

    def _generate_topics(self, sample_text: str, parent_topic: str, depth: int) -> List[str]:
        """Generate subtopics using LLM."""
        specificity = ["broad main", "specific", "very detailed"][min(depth, 2)]

        prompt = f"""Given the following document text, generate {self.tree_degree} {specificity} subtopics related to "{parent_topic}".

Document sample:
{sample_text[:2000]}

Provide exactly {self.tree_degree} subtopics as a JSON list. Example: ["Topic A", "Topic B", "Topic C"]

Output only the JSON list, no other text:"""

        for attempt in range(3):  # Retry up to 3 times
            if attempt > 0:
                time.sleep(min(2 ** attempt, 16))
            try:
                response = self.client.chat.completions.create(
                    model=self.config.topic_model,
                    messages=[
                        {"role": "system", "content": "You are a topic extraction assistant. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )

                content = response.choices[0].message.content

                # Extract JSON
                import json
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

                topics = json.loads(content.strip())
                if isinstance(topics, list) and len(topics) > 0:
                    return [str(t) for t in topics]
            except Exception:
                continue

        # Fallback: return generic topics
        return [f"{parent_topic} - Aspect {i+1}" for i in range(self.tree_degree)]

    def _get_node_path(self, root: TopicNode, target: TopicNode) -> List[str]:
        """Get path from root to target node."""
        if root == target:
            return [root.name]

        for child in root.children:
            if child == target:
                return [root.name, child.name]
            path = self._get_node_path(child, target)
            if path:
                return [root.name] + path

        return []

    def _collect_leaf_paths(self, node: TopicNode) -> List[List[str]]:
        """Collect all root-to-leaf paths."""
        if not node.children:
            return [[node.name]]

        paths = []
        for child in node.children:
            for child_path in self._collect_leaf_paths(child):
                paths.append([node.name] + child_path)

        return paths if paths else [[node.name]]

    def get_next_topic_path(self) -> Optional[List[str]]:
        """Get topic path for next iteration."""
        if not self.leaf_paths:
            return None

        # Cycle through leaf paths, going deeper with iterations
        idx = self.current_iteration % len(self.leaf_paths)
        path = self.leaf_paths[idx]

        # As iterations increase, use more specific (deeper) topics
        depth_to_use = min(self.current_iteration // 5 + 1, len(path))

        self.current_iteration += 1
        return path[:depth_to_use]

    def get_topic_context(self, path: List[str]) -> str:
        """Get topic context string for a path."""
        return " -> ".join(path)

    def visualize(self, logger=None) -> str:
        """
        Return ASCII tree visualization of the topic structure.
        
        Args:
            logger: Optional logger to print directly
            
        Returns:
            String representation of the tree
        """
        if not self.root:
            return "No tree built yet."
        
        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("TOPIC TREE STRUCTURE")
        lines.append("=" * 50)
        
        self._visualize_node(self.root, "", True, lines)
        
        lines.append("=" * 50)
        lines.append(f"Total leaf paths: {len(self.leaf_paths)}")
        lines.append("")
        
        result = "\n".join(lines)
        
        if logger:
            logger.info(result)
        
        return result
    
    def _visualize_node(self, node: TopicNode, prefix: str, is_last: bool, lines: List[str]) -> None:
        """Recursively build tree visualization."""
        # Current node line
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node.name}")
        
        # Process children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            self._visualize_node(child, child_prefix, is_last_child, lines)
