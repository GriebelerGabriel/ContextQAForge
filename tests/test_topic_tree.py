"""Tests for the topic_tree module."""

import json
from unittest.mock import MagicMock, patch

from topic_tree import TopicTree, TopicNode
from config import PipelineConfig


class TestTopicNode:
    def test_add_child(self):
        parent = TopicNode("Root", depth=0)
        child = TopicNode("Child", depth=1)
        parent.add_child(child)
        assert len(parent.children) == 1
        assert child.name == "Child"


class TestTopicTree:
    def test_init(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=3, tree_depth=2)
        assert tree.tree_degree == 3
        assert tree.tree_depth == 2
        assert tree.root is None

    def test_collect_leaf_paths(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        # Manually build a small tree
        root = TopicNode("Root", depth=0)
        child1 = TopicNode("A", depth=1)
        child2 = TopicNode("B", depth=1)
        root.add_child(child1)
        root.add_child(child2)

        tree.root = root
        tree.leaf_paths = tree._collect_leaf_paths(root)

        assert len(tree.leaf_paths) == 2
        assert ["Root", "A"] in tree.leaf_paths
        assert ["Root", "B"] in tree.leaf_paths

    def test_get_next_topic_path_cycles(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        root = TopicNode("Root", depth=0)
        child1 = TopicNode("A", depth=1)
        child2 = TopicNode("B", depth=1)
        root.add_child(child1)
        root.add_child(child2)

        tree.root = root
        tree.leaf_paths = tree._collect_leaf_paths(root)

        # First call returns first leaf
        path1 = tree.get_next_topic_path()
        assert path1 is not None
        # Second call returns second leaf
        path2 = tree.get_next_topic_path()
        assert path2 is not None
        # Third call cycles back
        path3 = tree.get_next_topic_path()
        assert path3 is not None

    def test_get_next_topic_path_empty(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        assert tree.get_next_topic_path() is None

    def test_get_topic_context(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        result = tree.get_topic_context(["Root", "Child", "Grandchild"])
        assert result == "Root -> Child -> Grandchild"

    def test_build_from_chunks(self, mock_config):
        """Test tree building with mocked LLM responses."""
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=1)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(["Topic A", "Topic B"])

        with patch.object(tree.client.chat.completions, "create", return_value=mock_response):
            chunks = ["Some content about topic A and topic B."]
            tree.build_from_chunks(chunks)

        assert tree.root is not None
        assert len(tree.leaf_paths) > 0

    def test_visualize_no_tree(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        result = tree.visualize()
        assert "No tree built" in result

    def test_visualize_with_tree(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        root = TopicNode("Root", depth=0)
        root.add_child(TopicNode("Child", depth=1))
        tree.root = root
        tree.leaf_paths = tree._collect_leaf_paths(root)

        result = tree.visualize()
        assert "Root" in result
        assert "Child" in result

    def test_find_relevant_chunks(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        chunks = [
            "Python is a programming language",
            "Java is also a programming language",
            "Cooking recipes for dinner",
        ]
        result = tree._find_relevant_chunks("Python", chunks)
        assert len(result) > 0
        assert "Python" in result[0]

    def test_find_relevant_chunks_no_match(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        chunks = ["Python is a language", "Java is also a language"]
        result = tree._find_relevant_chunks("Cooking", chunks)
        assert len(result) == 0

    def test_get_node_path(self, mock_config):
        tree = TopicTree(config=mock_config, tree_degree=2, tree_depth=2)
        root = TopicNode("Root", depth=0)
        child = TopicNode("Child", depth=1)
        grandchild = TopicNode("Grandchild", depth=2)
        root.add_child(child)
        child.add_child(grandchild)
        tree.root = root

        path = tree._get_node_path(root, grandchild)
        assert path == ["Root", "Child", "Grandchild"]
