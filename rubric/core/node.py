"""Rubric node implementation for the tree-based rubric system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import warnings

if TYPE_CHECKING:
    from .scorer import LeafScorer


@dataclass
class RubricNode:
    """A node in the rubric tree representing a criterion.

    Each node can be either a parent node (with children) or a leaf node (with a scorer).
    Nodes can be marked as critical or non-critical, affecting parent score computation.
    """

    name: str
    description: str
    is_critical: bool = False
    children: List[RubricNode] = field(default_factory=list)
    scorer: Optional["LeafScorer"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _score: Optional[float] = field(default=None, init=False)
    _reason: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate node configuration after initialization."""
        if self.children and self.scorer:
            raise ValueError("Node cannot have both children and a scorer")
        if not self.children and not self.scorer:
            raise ValueError("Node must have either children or a scorer")

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    @property
    def is_parent(self) -> bool:
        """Check if this is a parent node."""
        return len(self.children) > 0

    def add_child(self, child: RubricNode) -> None:
        """Add a child node.

        Args:
            child: The child node to add.

        Raises:
            ValueError: If this node already has a scorer.
        """
        if self.scorer:
            raise ValueError("Cannot add children to a node with a scorer")
        self.children.append(child)

    def remove_child(self, child: RubricNode) -> None:
        """Remove a child node.

        Args:
            child: The child node to remove.
        """
        if child in self.children:
            self.children.remove(child)

    def set_scorer(self, scorer: LeafScorer) -> None:
        """Set the scorer for this leaf node.

        Args:
            scorer: The scorer to use for this leaf node.

        Raises:
            ValueError: If this node has children.
        """
        if self.children:
            raise ValueError("Cannot set scorer on a node with children")
        self.scorer = scorer

    def get_critical_children(self) -> List[RubricNode]:
        """Get all critical child nodes."""
        return [child for child in self.children if child.is_critical]

    def get_non_critical_children(self) -> List[RubricNode]:
        """Get all non-critical child nodes."""
        return [child for child in self.children if not child.is_critical]

    def has_critical_children(self) -> bool:
        """Check if this node has any critical children."""
        return any(child.is_critical for child in self.children)

    def has_non_critical_children(self) -> bool:
        """Check if this node has any non-critical children."""
        return any(not child.is_critical for child in self.children)

    def _generate_parent_reason(self) -> str:
        """Generate a reason for a parent node based on its children's scores and reasons.
        
        Args:
            context: Context data for scoring.
            
        Returns:
            A generated reason explaining the parent node's score.
        """
        from ..utils.llm_client import create_llm_client
        
        # Prepare information about children for LLM
        children_info = []
        for child in self.children:
            child_info = {
                "name": child.name,
                "description": child.description,
                "is_critical": child.is_critical,
                "score": child.score,
                "reason": child.reason if child.reason else "No reason available"
            }
            children_info.append(child_info)
        
        # Create prompt for LLM
        prompt = f"""You are evaluating a rubric criterion called "{self.name}": {self.description}

This criterion has the following sub-criteria with their scores and reasons:

"""
        
        for child_info in children_info:
            critical_label = "CRITICAL" if child_info["is_critical"] else "NON-CRITICAL"
            prompt += f"- {child_info['name']} ({critical_label}): Score {child_info['score']:.2f}\n"
            prompt += f"  Description: {child_info['description']}\n"
            prompt += f"  Reason: {child_info['reason']}\n\n"
        
        prompt += f"""
The overall score for "{self.name}" is {self._score:.2f}.

Rubric scoring rules:
- Score is 0 if any critical child has score 0
- Score is average of non-critical children if all critical children have score 1
- Score is average of all children if no critical children exist or mixed performance

Please provide a concise reason (1-5 sentences) explaining why this criterion received a score of {self._score:.2f}, referencing the relevant sub-criteria and their performance. Focus on the most important factors that determined the score.
Make the the reason more natural language and human-like rather than formulaic, and avoid including numerical scores in the reasoning.
"""
        
        try:
            llm_client = create_llm_client()
            reason = llm_client.simple_completion(prompt, temperature=0.3)
            return reason.strip()
        except Exception as e:
            # Fallback to basic reason if LLM fails
            # Add warning that we are falling back
            warnings.warn(f"Failed to use LLM to generate reason for parent node {self.name}, reason: {e}, using simple fallback instead")
            return f"Score {self._score:.2f} based on performance across {len(self.children)} sub-criteria"

    def compute_score(self, **context) -> float:
        """Compute the score for this node.

        For leaf nodes, uses the scorer. For parent nodes, computes based on children
        according to the rubric rules:
        - Score is 0 if any critical child has score 0
        - Score is average of non-critical children if all critical children have score 1
        - Score is average of all children if no critical children exist

        Args:
            context: Context data for scoring.

        Returns:
            Tuple containing the reason for the score and the score between 0 and 1.
        """
        if self.is_leaf:
            if not self.scorer:
                raise ValueError("Leaf node must have a scorer")
            self._score, self._reason = self.scorer.score(**context)
            return self._score

        # Parent node scoring logic
        critical_children = self.get_critical_children()

        # Compute scores for all children and store them
        all_scores = []
        critical_scores = []
        non_critical_scores = []

        for child in self.children:
            score = child.compute_score(**context)
            all_scores.append(score)

            if child.is_critical:
                critical_scores.append(score)
            else:
                non_critical_scores.append(score)

        # Apply scoring rules
        if critical_children:
            # Check if any critical child has score 0
            if any(score == 0 for score in critical_scores):
                self._score = 0.0
            # If all critical children have score 1, use non-critical average
            elif all(score == 1 for score in critical_scores):
                if non_critical_scores:
                    self._score = sum(non_critical_scores) / len(non_critical_scores)
                else:
                    self._score = 1.0
            # Otherwise, use average of all children
            else:
                self._score = sum(all_scores) / len(all_scores)
        else:
            # No critical children - use average of all children
            if all_scores:
                self._score = sum(all_scores) / len(all_scores)
            else:
                self._score = 0.0
        
        return self._score

    @property
    def score(self) -> float:
        """Get the last computed score for this node."""
        # TODO: Maybe 0.0 is not the best default score
        return self._score if self._score is not None else 0.0

    @property
    def reason(self) -> str:
        """Get the reason for the last computed score for this node."""
        if self._reason is None and not self.is_leaf:
            self._reason = self._generate_parent_reason()
        return self._reason if self._reason is not None else "No reason available yet"

    def reset_scores(self) -> None:
        """Reset scores for this node and all descendants."""
        self._score = None
        self._reason = None
        for child in self.children:
            child.reset_scores()

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        result = {
            "name": self.name,
            "description": self.description,
            "is_critical": self.is_critical,
            "metadata": self.metadata,
        }

        if self.is_leaf and self.scorer:
            result["scorer"] = self.scorer.to_dict()
        else:
            result["children"] = [child.to_dict() for child in self.children]

        if self._score is not None:
            result["score"] = self._score

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricNode:
        """Create node from dictionary representation.

        Args:
            data: Dictionary containing node data.

        Returns:
            RubricNode instance.
        """
        from .scorer import LeafScorer

        # Create node with minimal data first
        if "scorer" in data:
            # Leaf node
            scorer = LeafScorer.from_dict(data["scorer"])
            node = cls(
                name=data["name"],
                description=data["description"],
                is_critical=data.get("is_critical", False),
                scorer=scorer,
                metadata=data.get("metadata", {}),
            )
        elif "children" in data:
            # Parent node - create children first
            children = [cls.from_dict(child_data) for child_data in data["children"]]
            node = cls(
                name=data["name"],
                description=data["description"],
                is_critical=data.get("is_critical", False),
                children=children,
                metadata=data.get("metadata", {}),
            )
        else:
            raise ValueError(f"Node '{data['name']}' must have either children or scorer")

        return node

    def __str__(self) -> str:
        """String representation of the node."""
        node_type = "LEAF" if self.is_leaf else "PARENT"
        critical = "CRITICAL" if self.is_critical else "NON-CRITICAL"
        score_str = f" (score: {self._score})" if self._score is not None else ""
        return f"{node_type} {critical}: {self.name}{score_str}"

    def __repr__(self) -> str:
        """Detailed string representation of the node."""
        return (
            f"RubricNode(name='{self.name}', is_critical={self.is_critical}, "
            f"children={len(self.children)}, scorer={self.scorer is not None})"
        )
