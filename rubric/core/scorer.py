"""Scoring implementations for leaf nodes in the rubric tree."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

SCORER_REGISTRY: dict[str, type[LeafScorer]] = {}

def register(scorer_type: str):
    """Register a scorer class.
    
    Args:
        scorer_type: Type of scorer.
    
    Returns:
        Decorator function that registers the class.
    """
    def decorator(scorer_class: type[LeafScorer]) -> type[LeafScorer]:
        SCORER_REGISTRY[scorer_type] = scorer_class
        return scorer_class
    return decorator

class LeafScorer(ABC):
    """Abstract base class for leaf node scorers."""

    @abstractmethod
    def score(self, **context) -> tuple[float, str]:
        """Compute score for the leaf node.

        Args:
            context: Context data for scoring.

        Returns:
            Tuple containing the reason for the score and the score between 0 and 1.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert scorer to dictionary representation."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> LeafScorer:
        """Create scorer from dictionary representation."""
        scorer_type = data.get("type")

        if scorer_type not in SCORER_REGISTRY:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")
        return SCORER_REGISTRY[scorer_type].from_dict(data)

    @classmethod
    @abstractmethod
    def get_json_description(cls) -> str:
        """Get the JSON format description for the scorer."""
        pass

@register("function")
class FunctionScorer(LeafScorer):
    """Scorer that uses a Python function to compute the score.

    The function should accept context data and return a score between 0 and 1.
    """

    def __init__(self, function_code: str):
        """Initialize FunctionScorer with function code.
        
        Args:
            function_code: Python function code that will be cleaned automatically.
        """
        self.function_code = function_code

    def _clean_function_code(self, code: str) -> str:
        """Clean function code by extracting from python code blocks if present.
        
        Args:
            code: Raw function code string.
            
        Returns:
            Cleaned function code string.
        """
        # Check if code is wrapped in ```python...``` block
        if code.strip().startswith("```python") and code.strip().endswith("```"):
            # Extract content between ```python and ```
            lines = code.strip().split('\n')
            # Remove first line (```python) and last line (```)
            content_lines = lines[1:-1]
            return '\n'.join(content_lines)
        else:
            # Return as-is if not in a code block
            return code

    @property
    def function_code(self) -> str:
        """Get the function code."""
        return self._function_code

    @function_code.setter
    def function_code(self, value: str) -> None:
        """Set the function code, cleaning it if necessary."""
        self._function_code = self._clean_function_code(value)

    def score(self, **global_context: Any) -> tuple[float, str]:
        """Execute the function to compute the score.

        Args:
            context: Context data passed to the function.

        Returns:
            Score between 0 and 1.

        Raises:
            ValueError: If function execution fails or returns invalid score.
        """
        try:
            # Create a namespace for the function
            namespace: dict[str, Any] = {}

            # Execute the function code
            exec(self.function_code, global_context, namespace)

            score_func = namespace["compute_score"]

            # Call the function
            reason, score = score_func()

            if not isinstance(reason, str) or not isinstance(score, (int, float)):
                raise ValueError(f"Function must return a string and a number, got {type(reason)} and {type(score)}")

            if not (0 <= score <= 1):
                raise ValueError(f"Score must be between 0 and 1, got {score}")

            return score, reason

        except Exception as e:
            raise ValueError(f"Function scoring failed: {str(e)}") from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorer to dictionary representation."""
        return {
            "type": "function",
            "function_code": self.function_code,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FunctionScorer:
        """Create scorer from dictionary representation."""
        if data.get("type") != "function":
            raise ValueError(f"Invalid scorer type: {data.get('type')}")

        return cls(
            function_code=data["function_code"],
        )

    @classmethod
    def get_json_description(cls) -> str:
        """Get the JSON format description for the scorer."""
        return """```json
        {
            "type": "function",
            "function_code": "```python\ndef compute_score() -> tuple[str, float]:\n    ...\n    return \"<REASON_FOR_SCORE>\", <SCORE> # The score should be between 0 and 1.\n```"
        }
        ```"""
