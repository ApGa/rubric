"""Rubric tree implementation for managing and evaluating rubric hierarchies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from rubric.utils.llm_client import LLMClient, create_llm_client
from rubric.utils.prompt_retriever import PromptRetriever

from .node import RubricNode


@dataclass
class RubricTree:
    """A tree structure for managing rubric evaluation criteria.

    The tree consists of RubricNode instances organized in a hierarchy,
    with a single root node representing the overall evaluation criterion.
    """

    root: RubricNode
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tree structure after initialization."""
        if not isinstance(self.root, RubricNode):
            raise ValueError("Root must be a RubricNode instance")

    def evaluate(
        self,
        include_reason: bool = False,
        compute_strategy: Literal["default", "mind2web2"] = "default",
        non_critical_weight: float = 0.3,
        **context: Any,
    ) -> tuple[float, str]:
        """Evaluate the entire rubric tree and return the overall score.

        Args:
            include_reason: Whether to include the reason for the score.
            compute_strategy: How parent nodes aggregate child scores
                ("default" or "mind2web2").
            non_critical_weight: Lambda (λ) used by the default strategy when
                mixing critical and non-critical children.
            context: Context data for evaluation.

        Returns:
            Overall score between 0 and 1. If include_reason is True, returns a tuple of the score
            and the reason.
        """
        self.root.compute_score(
            compute_strategy=compute_strategy,
            non_critical_weight=non_critical_weight,
            **context,
        )
        if include_reason:
            return self.root.score, self.root.reason
        else:
            return (self.root.score, "")

    def reset_scores(self) -> None:
        """Reset all scores in the tree."""
        self.root.reset_scores()

    def get_all_nodes(self) -> List[RubricNode]:
        """Get all nodes in the tree in depth-first order.

        Returns:
            List of all nodes in the tree.
        """
        nodes: List[RubricNode] = []
        self._collect_nodes(self.root, nodes)
        return nodes

    def _collect_nodes(self, node: RubricNode, nodes: List[RubricNode]) -> None:
        """Recursively collect all nodes in depth-first order."""
        nodes.append(node)
        for child in node.children:
            self._collect_nodes(child, nodes)

    def get_leaf_nodes(self) -> List[RubricNode]:
        """Get all leaf nodes in the tree.

        Returns:
            List of leaf nodes.
        """
        return [node for node in self.get_all_nodes() if node.is_leaf]

    def get_parent_nodes(self) -> List[RubricNode]:
        """Get all parent nodes in the tree.

        Returns:
            List of parent nodes.
        """
        return [node for node in self.get_all_nodes() if node.is_parent]

    def get_critical_nodes(self) -> List[RubricNode]:
        """Get all critical nodes in the tree.

        Returns:
            List of critical nodes.
        """
        return [node for node in self.get_all_nodes() if node.is_critical]

    def find_node_by_name(self, name: str) -> Optional[RubricNode]:
        """Find a node by its name.

        Args:
            name: Name of the node to find.

        Returns:
            The node if found, None otherwise.
        """
        for node in self.get_all_nodes():
            if node.name == name:
                return node
        return None

    def find_nodes_by_criteria(self, **criteria: Any) -> List[RubricNode]:
        """Find nodes matching the given criteria.

        Args:
            **criteria: Criteria to match (e.g., is_critical=True).

        Returns:
            List of matching nodes.
        """
        matching_nodes = []
        for node in self.get_all_nodes():
            match = True
            for key, value in criteria.items():
                if not hasattr(node, key) or getattr(node, key) != value:
                    match = False
                    break
            if match:
                matching_nodes.append(node)
        return matching_nodes

    def get_tree_depth(self) -> int:
        """Get the maximum depth of the tree.

        Returns:
            Maximum depth (root is depth 0).
        """
        return self._get_node_depth(self.root)

    def _get_node_depth(self, node: RubricNode) -> int:
        """Get the maximum depth starting from a node."""
        if node.is_leaf:
            return 0
        return 1 + max(self._get_node_depth(child) for child in node.children)

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the tree structure.

        Returns:
            Dictionary with tree statistics.
        """
        all_nodes = self.get_all_nodes()
        leaf_nodes = self.get_leaf_nodes()
        critical_nodes = self.get_critical_nodes()

        return {
            "total_nodes": len(all_nodes),
            "leaf_nodes": len(leaf_nodes),
            "parent_nodes": len(all_nodes) - len(leaf_nodes),
            "critical_nodes": len(critical_nodes),
            "non_critical_nodes": len(all_nodes) - len(critical_nodes),
            "max_depth": self.get_tree_depth(),
        }

    def print_tree(self, show_scores: bool = True) -> None:
        """Print a visual representation of the tree.

        Args:
            show_scores: Whether to show computed scores.
        """
        self._print_node(self.root, "", True, show_scores)

    def _print_node(self, node: RubricNode, prefix: str, is_last: bool, show_scores: bool) -> None:
        """Recursively print nodes with tree structure."""
        # Create the current line
        connector = "└── " if is_last else "├── "
        node_str = (
            str(node)
            if show_scores
            else f"{node.name} ({'CRITICAL' if node.is_critical else 'NON-CRITICAL'})"
        )
        print(f"{prefix}{connector}{node_str}")

        # Print children
        if node.children:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension

            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                self._print_node(child, new_prefix, is_last_child, show_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation.

        Returns:
            Dictionary representation of the tree.
        """
        return {
            "root": self.root.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricTree:
        """Create tree from dictionary representation.

        Args:
            data: Dictionary containing tree data.

        Returns:
            RubricTree instance.
        """
        root = RubricNode.from_dict(data["root"])
        metadata = data.get("metadata", {})
        return cls(root=root, metadata=metadata)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save tree to a JSON file.

        Args:
            file_path: Path to save the file.
        """
        file_path = Path(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_as_dir(self, dir_path: Union[str, Path]) -> None:
        """Save the entire tree to a directory for human-friendly editing.

        Layout:
        - metadata.json: tree-level metadata
        - root/: directory for the root node (and recursively children)
        """
        dir_p = Path(dir_path)
        dir_p.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(dir_p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Save root node recursively
        root_dir = dir_p / "root"
        self.root.save_as_dir(root_dir)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> RubricTree:
        """Load tree from a JSON file.

        Args:
            file_path: Path to the file.

        Returns:
            RubricTree instance.
        """
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_from_dir(cls, dir_path: Union[str, Path]) -> "RubricTree":
        """Load a tree previously saved via `save_as_dir`."""
        dir_p = Path(dir_path)
        meta_path = dir_p / "metadata.json"
        root_dir = dir_p / "root"

        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        if not root_dir.exists():
            raise FileNotFoundError(f"Missing root directory in {dir_p}")

        from .node import RubricNode

        root = RubricNode.load_from_dir(root_dir)
        return cls(root=root, metadata=metadata)

    def validate_tree(self) -> List[str]:
        """Validate the tree structure and return any issues found.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check all nodes
        for node in self.get_all_nodes():
            # Validate leaf nodes have scorers
            if node.is_leaf and not node.scorer:
                errors.append(f"Leaf node '{node.name}' has no scorer")

            # Validate parent nodes have children
            if node.is_parent and not node.children:
                errors.append(f"Parent node '{node.name}' has no children")

            # Validate node names are unique
            all_nodes = self.get_all_nodes()
            names = [n.name for n in all_nodes]
            if names.count(node.name) > 1:
                errors.append(f"Duplicate node name: '{node.name}'")

        return errors

    def is_valid(self) -> bool:
        """Check if the tree is valid.

        Returns:
            True if tree is valid, False otherwise.
        """
        return len(self.validate_tree()) == 0

    def get_evaluation_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get a detailed evaluation report for the tree.

        Args:
            context: Context data for evaluation.

        Returns:
            Detailed evaluation report.
        """
        # Evaluate the tree
        overall_score = self.evaluate(**context)

        # Collect node scores
        node_scores = {}
        for node in self.get_all_nodes():
            if node.score is not None:
                node_scores[node.name] = {
                    "score": node.score,
                    "is_critical": node.is_critical,
                    "is_leaf": node.is_leaf,
                    "description": node.description,
                }

        # Get tree statistics
        stats = self.get_tree_stats()

        return {
            "overall_score": overall_score,
            "node_scores": node_scores,
            "tree_stats": stats,
            "evaluation_context": context,
        }

    def __str__(self) -> str:
        """String representation of the tree."""
        stats = self.get_tree_stats()
        return (
            f"RubricTree(root='{self.root.name}', "
            f"nodes={stats['total_nodes']}, "
            f"depth={stats['max_depth']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the tree."""
        return str(self)

    def visualize(
        self,
        method: str = "plotly",
        show_scores: bool = False,
        layout: str = "hierarchical",
        width: int = 1600,
        height: int = 1000,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Visualize the rubric tree interactively.

        Args:
            method: Visualization method ('plotly', 'plotly_network', 'html').
            show_scores: Whether to show scores in the visualization.
            layout: Layout for network plots ('hierarchical', 'circular', 'spring').
            width: Width of the plot in pixels.
            height: Height of the plot in pixels.
            title: Title for the visualization.
            **kwargs: Additional arguments passed to the visualizer.

        Returns:
            Plotly figure object for plotly methods, HTML path for html method.
        """
        from ..utils.visualizer import RubricTreeVisualizer

        if title is None:
            title = f"Rubric Tree: {self.root.name}"

        visualizer = RubricTreeVisualizer()

        if method == "plotly":
            fig = visualizer.visualize_tree_plotly(
                tree=self,
                show_scores=show_scores,
                width=width,
                height=height,
                title=title,
            )
            if fig:
                fig.show()
            return fig

        elif method == "plotly_network":
            fig = visualizer.visualize_tree_plotly(
                tree=self,
                show_scores=show_scores,
                layout=layout,
                width=width,
                height=height,
                title=title,
            )
            if fig:
                fig.show()
            return fig

        elif method == "html":
            # Fallback to HTML file generation
            output_path = kwargs.get("output_path")
            return visualizer.generate_interactive_html(
                tree=self,
                file_path=output_path,
                show_scores=show_scores,
            )
        else:
            raise ValueError(f"Unknown visualization method: {method}")

    def plot(
        self,
        show_scores: bool = False,
        layout: str = "hierarchical",
        width: int = 1600,
        height: int = 1000,
        title: Optional[str] = None,
    ) -> Optional[Any]:
        """Quick plot method for interactive tree visualization.

        Args:
            show_scores: Whether to show scores.
            layout: Layout algorithm ('hierarchical', 'circular', 'spring').
            width: Width of the plot.
            height: Height of the plot.
            title: Title for the plot.

        Returns:
            Plotly figure object.
        """
        return self.visualize(
            method="plotly",
            show_scores=show_scores,
            layout=layout,
            width=width,
            height=height,
            title=title,
        )

    def plot_network(
        self,
        show_scores: bool = True,
        layout: str = "hierarchical",
        width: int = 1600,
        height: int = 1000,
        title: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot as network-style tree visualization.

        Args:
            show_scores: Whether to show scores.
            layout: Layout algorithm ('hierarchical', 'circular', 'spring').
            width: Width of the plot.
            height: Height of the plot.
            title: Title for the plot.

        Returns:
            Plotly figure object.
        """
        return self.visualize(
            method="plotly_network",
            show_scores=show_scores,
            layout=layout,
            width=width,
            height=height,
            title=title,
        )

    def generate_text_tree(self, show_scores: bool = True, max_width: int = 100) -> str:
        """Generate a text-based tree representation.

        Args:
            show_scores: Whether to include scores in the output.
            max_width: Maximum width for wrapping descriptions.

        Returns:
            Text representation of the tree.
        """
        lines: List[str] = []
        self._generate_text_node(self.root, "", True, show_scores, max_width, lines)
        return "\n".join(lines)

    def _generate_text_node(
        self,
        node: RubricNode,
        prefix: str,
        is_last: bool,
        show_scores: bool,
        max_width: int,
        lines: List[str],
    ) -> None:
        """Recursively generate text representation of nodes."""
        import textwrap

        # Create the current line
        connector = "└── " if is_last else "├── "

        # Build node info
        critical_marker = " ⚠️" if node.is_critical else ""
        node_type = "🍃" if node.is_leaf else "📁"
        score_str = f" [{node.score:.2f}]" if show_scores and node.score is not None else ""

        node_info = f"{node_type} {node.name}{critical_marker}{score_str}"
        lines.append(f"{prefix}{connector}{node_info}")

        # Add description if it's not too long
        if node.description and len(node.description) <= max_width:
            desc_prefix = "    " if is_last else "│   "
            wrapped_desc = textwrap.fill(
                node.description,
                width=max_width - len(prefix) - len(desc_prefix) - 4,
                initial_indent="    ",
                subsequent_indent="    ",
            )
            lines.append(f"{prefix}{desc_prefix}💬 {wrapped_desc.strip()}")

        # Add children
        if node.children:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension

            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                self._generate_text_node(
                    child, new_prefix, is_last_child, show_scores, max_width, lines
                )

    @property
    def score(self) -> float:
        """Get the overall score of the tree."""
        return self.root.score

    @property
    def reason(self) -> str:
        """Get the reason for the overall score of the tree."""
        return self.root.reason

    @classmethod
    def generate(
        cls,
        task: str,
        llm_client: LLMClient | None = None,
        prompt_retriever: PromptRetriever | None = None,
        compute_strategy: Literal["default", "mind2web2"] = "default",
        non_critical_weight: float = 0.7,
        **kwargs: Any,
    ) -> RubricTree:
        """Generate a rubric tree for a task.

        Args:
            task: Description of the task to create a rubric for.
            llm_client: Optional LLM client instance.
            prompt_retriever: Optional prompt retriever.
            compute_strategy: How parent nodes aggregate child scores
                ("default" or "mind2web2").
            non_critical_weight: λ in overall = max(0, avg(critical) − λ * (1 − avg(non-critical)))
            **kwargs: Additional arguments forwarded to the underlying generator.

        Returns:
            Generated RubricTree.
        """
        from ..generate.tree_generator import RubricTreeGenerator

        llm_client = llm_client or create_llm_client()
        prompt_retriever = prompt_retriever or PromptRetriever()

        generator = RubricTreeGenerator(llm_client=llm_client, prompt_retriever=prompt_retriever)

        return generator.generate_rubric_tree(
            task,
            compute_strategy=compute_strategy,
            non_critical_weight=non_critical_weight,
            **kwargs,
        )
