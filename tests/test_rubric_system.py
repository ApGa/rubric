"""Tests for the rubric system components."""

import tempfile
from pathlib import Path

import pytest

from rubric.core import RubricNode, RubricTree
from rubric.core.scorer import FunctionScorer, ScriptScorer
from rubric.examples.basic_examples import (
    create_code_review_rubric,
    create_simple_rubric,
)


class TestRubricNode:
    """Test RubricNode functionality."""

    def test_leaf_node_creation(self) -> None:
        """Test creating a leaf node with a scorer."""
        scorer = FunctionScorer(
            function_code="def score_function(context): return 0.8", function_name="score_function"
        )

        node = RubricNode(
            name="Test Node", description="A test node", is_critical=True, scorer=scorer
        )

        assert node.name == "Test Node"
        assert node.description == "A test node"
        assert node.is_critical is True
        assert node.is_leaf is True
        assert node.is_parent is False
        assert node.scorer == scorer

    def test_parent_node_creation(self) -> None:
        """Test creating a parent node with children."""
        child1 = RubricNode(
            name="Child 1",
            description="First child",
            scorer=FunctionScorer("def score_function(context): return 1.0"),
        )

        child2 = RubricNode(
            name="Child 2",
            description="Second child",
            scorer=FunctionScorer("def score_function(context): return 0.5"),
        )

        parent = RubricNode(name="Parent", description="Parent node", children=[child1, child2])

        assert parent.is_parent is True
        assert parent.is_leaf is False
        assert len(parent.children) == 2
        assert child1 in parent.children
        assert child2 in parent.children

    def test_invalid_node_creation(self) -> None:
        """Test that invalid node configurations raise errors."""
        scorer = FunctionScorer("def score_function(context): return 1.0")
        child = RubricNode("Child", "Child node", scorer=scorer)

        # Node with both children and scorer should fail
        with pytest.raises(ValueError):
            RubricNode(name="Invalid", description="Invalid node", children=[child], scorer=scorer)

        # Node with neither children nor scorer should fail
        with pytest.raises(ValueError):
            RubricNode(name="Invalid", description="Invalid node")

    def test_scoring_rules(self) -> None:
        """Test the scoring rules for parent nodes."""
        # Create leaf nodes
        critical_pass = RubricNode(
            name="Critical Pass",
            description="Critical node that passes",
            is_critical=True,
            scorer=FunctionScorer("def score_function(context): return 1.0"),
        )

        critical_fail = RubricNode(
            name="Critical Fail",
            description="Critical node that fails",
            is_critical=True,
            scorer=FunctionScorer("def score_function(context): return 0.0"),
        )

        non_critical = RubricNode(
            name="Non-Critical",
            description="Non-critical node",
            is_critical=False,
            scorer=FunctionScorer("def score_function(context): return 0.8"),
        )

        # Test: Critical child fails -> parent score is 0
        parent1 = RubricNode(
            name="Parent 1",
            description="Parent with failing critical child",
            children=[critical_fail, non_critical],
        )

        score1 = parent1.compute_score({})
        assert score1 == 0.0

        # Test: All critical children pass -> use non-critical average
        parent2 = RubricNode(
            name="Parent 2",
            description="Parent with passing critical child",
            children=[critical_pass, non_critical],
        )

        score2 = parent2.compute_score({})
        assert score2 == 0.8  # Only non-critical score

        # Test: No critical children -> use average of all
        parent3 = RubricNode(
            name="Parent 3",
            description="Parent with only non-critical children",
            children=[
                non_critical,
                RubricNode(
                    "Non-Critical 2",
                    "Another non-critical",
                    scorer=FunctionScorer("def score_function(context): return 0.6"),
                ),
            ],
        )

        score3 = parent3.compute_score({})
        assert score3 == 0.7  # (0.8 + 0.6) / 2

    def test_serialization(self) -> None:
        """Test node serialization to/from dict."""
        scorer = FunctionScorer(
            function_code="def score_function(context): return 0.5", function_name="score_function"
        )

        node = RubricNode(
            name="Test Node",
            description="A test node",
            is_critical=True,
            scorer=scorer,
            metadata={"test": "value"},
        )

        # Convert to dict
        node_dict = node.to_dict()

        # Convert back to node
        restored_node = RubricNode.from_dict(node_dict)

        assert restored_node.name == node.name
        assert restored_node.description == node.description
        assert restored_node.is_critical == node.is_critical
        assert restored_node.metadata == node.metadata
        assert isinstance(restored_node.scorer, FunctionScorer)
        assert restored_node.scorer.function_code == scorer.function_code


class TestRubricTree:
    """Test RubricTree functionality."""

    def test_tree_creation(self) -> None:
        """Test creating a rubric tree."""
        tree = create_simple_rubric()

        assert tree.root is not None
        assert tree.root.name == "Essay Evaluation"
        assert len(tree.root.children) == 3

    def test_tree_evaluation(self) -> None:
        """Test evaluating a rubric tree."""
        # Create a simple tree for testing
        leaf1 = RubricNode(
            name="Leaf 1",
            description="First leaf",
            scorer=FunctionScorer("def score_function(context): return 0.8"),
        )

        leaf2 = RubricNode(
            name="Leaf 2",
            description="Second leaf",
            scorer=FunctionScorer("def score_function(context): return 0.6"),
        )

        root = RubricNode(name="Root", description="Root node", children=[leaf1, leaf2])

        tree = RubricTree(root=root)

        # Evaluate the tree
        score = tree.evaluate({})
        assert score == 0.7  # (0.8 + 0.6) / 2

    def test_tree_serialization(self) -> None:
        """Test tree serialization to/from file."""
        tree = create_simple_rubric()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            tree.save_to_file(temp_path)

            # Load from file
            loaded_tree = RubricTree.load_from_file(temp_path)

            assert loaded_tree.root.name == tree.root.name
            assert len(loaded_tree.root.children) == len(tree.root.children)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_tree_validation(self) -> None:
        """Test tree validation."""
        # Valid tree
        valid_tree = create_simple_rubric()
        assert valid_tree.is_valid()
        assert len(valid_tree.validate_tree()) == 0

        # Create an invalid tree by manually breaking a valid one
        invalid_tree = create_simple_rubric()
        # Remove scorer from a leaf node to make it invalid
        leaf_node = invalid_tree.get_leaf_nodes()[0]
        leaf_node.scorer = None

        assert not invalid_tree.is_valid()
        errors = invalid_tree.validate_tree()
        assert len(errors) > 0
        assert "no scorer" in errors[0].lower()

    def test_tree_stats(self) -> None:
        """Test tree statistics."""
        tree = create_simple_rubric()
        stats = tree.get_tree_stats()

        assert stats["total_nodes"] == 4  # 1 root + 3 children
        assert stats["leaf_nodes"] == 3
        assert stats["parent_nodes"] == 1
        assert stats["max_depth"] == 1


class TestScorers:
    """Test different scorer implementations."""

    def test_function_scorer(self) -> None:
        """Test FunctionScorer."""
        scorer = FunctionScorer(
            function_code="""
def score_function(context):
    value = context.get('value', 0)
    return min(1.0, value / 10.0)
""",
            function_name="score_function",
        )

        # Test scoring
        score1 = scorer.score({"value": 5})
        assert score1 == 0.5

        score2 = scorer.score({"value": 15})
        assert score2 == 1.0

        # Test serialization
        scorer_dict = scorer.to_dict()
        restored_scorer = FunctionScorer.from_dict(scorer_dict)
        assert restored_scorer.function_code == scorer.function_code

    def test_script_scorer(self) -> None:
        """Test ScriptScorer."""
        scorer = ScriptScorer(
            script_content="""
import json
import sys

context = json.loads(sys.stdin.read())
value = context.get('value', 0)
score = min(1.0, value / 10.0)
print(score)
""",
            script_language="python",
        )

        # Test scoring
        score = scorer.score({"value": 7})
        assert score == 0.7

        # Test serialization
        scorer_dict = scorer.to_dict()
        restored_scorer = ScriptScorer.from_dict(scorer_dict)
        assert restored_scorer.script_content == scorer.script_content


class TestExamples:
    """Test the example rubrics."""

    def test_simple_rubric_structure(self) -> None:
        """Test the simple rubric structure."""
        tree = create_simple_rubric()

        assert tree.root.name == "Essay Evaluation"
        assert len(tree.root.children) == 3

        # Check child names
        child_names = [child.name for child in tree.root.children]
        assert "Grammar and Spelling" in child_names
        assert "Content Quality" in child_names
        assert "Writing Style" in child_names

        # Check critical designations
        critical_children = tree.root.get_critical_children()
        assert len(critical_children) == 2  # Grammar and Content are critical

    def test_code_review_rubric_structure(self) -> None:
        """Test the code review rubric structure."""
        tree = create_code_review_rubric()

        assert tree.root.name == "Code Review"
        assert len(tree.root.children) == 2

        # Check that it has functionality and quality branches
        child_names = [child.name for child in tree.root.children]
        assert "Functionality" in child_names
        assert "Code Quality" in child_names

    def test_code_review_evaluation(self) -> None:
        """Test evaluating the code review rubric."""
        tree = create_code_review_rubric()

        context = {
            "code": "def hello(): print('Hello, World!')",
            "test_results": {"passed": 10, "total": 10},
        }

        # This should work without LLM calls for the script scorer
        try:
            score = tree.evaluate(context)
            assert 0 <= score <= 1
        except Exception as e:
            # LLM scorers might fail without proper API setup
            assert "LLM" in str(e) or "API" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])
