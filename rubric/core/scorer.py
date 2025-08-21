"""Scoring implementations for leaf nodes in the rubric tree."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from rubric.utils.llm_tools import LLM_MODEL_NAME

SCORER_REGISTRY: dict[str, type[LeafScorer]] = {}


def register(scorer_type: str) -> Callable[[type[LeafScorer]], type[LeafScorer]]:
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
    def score(self, **context: Any) -> tuple[float, str]:
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

    @classmethod
    @abstractmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for configuring this scorer type."""
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
            lines = code.strip().split("\n")
            # Remove first line (```python) and last line (```)
            content_lines = lines[1:-1]
            return "\n".join(content_lines)
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
                raise ValueError(
                    f"Function must return a string and a number, got {type(reason)}"
                    f" and {type(score)}"
                )

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

        return (
            "```json\n"
            "        {\n"
            '            "type": "function",\n'
            '            "function_code": "```python\\n'
            "def compute_score() -> tuple[str, float]:\\n"
            "    ...\\n"
            '    return \\"<REASON_FOR_SCORE>\\", <SCORE> '
            '# The score should be between 0 and 1.\\n```"\n'
            "        }\n"
            "        ```"
        )

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["function"]},
                "function_code": {"type": "string"},
            },
            "required": ["type", "function_code"],
        }


@register("llm")
class LLMScorer(LeafScorer):
    """Scorer that uses an LLM to compute the score with custom prompts.

    This scorer sends a system prompt and user prompt to an LLM and expects
    the LLM to return a structured response with a score and reason.
    """

    def __init__(self, system_prompt: str, user_prompt: str):
        """Initialize LLMScorer with system and user prompts.

        Args:
            system_prompt: System prompt to set the context for the LLM.
            user_prompt: User prompt with the specific scoring request.
        """
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def score(self, **context: Any) -> tuple[float, str]:
        """Use LLM to compute the score.

        Args:
            context: Context data that can be used to format prompts.

        Returns:
            Tuple containing (score, reason) where score is between 0 and 1.

        Raises:
            ValueError: If LLM call fails or returns invalid response.
        """
        try:
            from ..utils.llm_client import create_llm_client

            # Format prompts with context if needed
            formatted_system_prompt = (
                self.system_prompt.format(**context) if context else self.system_prompt
            )
            formatted_user_prompt = (
                self.user_prompt.format(**context) if context else self.user_prompt
            )

            # Create LLM client and make request
            llm_client = create_llm_client(model=LLM_MODEL_NAME)
            response = llm_client.system_completion(
                system_prompt=formatted_system_prompt,
                user_prompt=formatted_user_prompt,
                temperature=0.3,  # Low temperature for consistent scoring
            )

            # Try to parse as JSON first (new structured format)
            try:
                # Look for JSON code block in the response
                import re

                # First try to find ```json code block
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE
                )

                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # Fallback: look for any ``` code block that might contain JSON
                    code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
                    if code_match:
                        json_str = code_match.group(1).strip()
                    else:
                        # Last resort: try the entire response as JSON
                        json_str = response.strip()

                # Parse the JSON response
                parsed_response = json.loads(json_str)

                # Extract score and reason from structured response
                if (
                    isinstance(parsed_response, dict)
                    and "score" in parsed_response
                    and "reason" in parsed_response
                ):
                    score = float(parsed_response["score"])
                    reason = str(parsed_response["reason"])

                    # Validate score range
                    if not (0 <= score <= 1):
                        raise ValueError(f"Score must be between 0 and 1, got {score}")

                    return score, reason
                else:
                    raise ValueError("JSON response missing required 'score' or 'reason' fields")

            except (json.JSONDecodeError, KeyError, ValueError):
                # Fall back to legacy parsing for backward compatibility
                # Parse the response - expect format like "Score: 0.85\nReason: ..."
                # or "Reason: ...\nScore: 0.85"
                lines = response.strip().split("\n")
                score = None
                reason_parts = []

                for line in lines:
                    line = line.strip()
                    if line.lower().startswith("score:"):
                        try:
                            score_str = line.split(":", 1)[1].strip()
                            score = float(score_str)
                        except (ValueError, IndexError):
                            continue
                    elif line.lower().startswith("reason:"):
                        reason_parts.append(line.split(":", 1)[1].strip())
                    elif line and not line.lower().startswith("score:"):
                        # Assume it's part of the reason if it's not a score line
                        reason_parts.append(line)

                # If we didn't find a structured response, try to extract from the end
                if score is None:
                    # Look for a number at the end that could be a score
                    import re

                    numbers = re.findall(r"\b0\.\d+\b|\b1\.0+\b|\b[01]\b", response)
                    if numbers:
                        try:
                            score = float(numbers[-1])
                            reason = response.rsplit(str(score), 1)[0].strip()
                            if not reason:
                                reason = "LLM provided score without detailed reasoning"
                        except ValueError:
                            pass

                if score is None:
                    raise ValueError(
                        f"Could not parse score from LLM response. Expected JSON format "
                        f'{{"reason": "...", "score": X.XX}} or legacy format. Got: {response}'
                    )

                reason = (
                    " ".join(reason_parts)
                    if reason_parts
                    else "LLM provided score without detailed reasoning"
                )

                # Validate score range
                if not (0 <= score <= 1):
                    raise ValueError(f"Score must be between 0 and 1, got {score}")

                return score, reason

        except Exception as e:
            raise ValueError(f"LLM scoring failed: {str(e)}") from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorer to dictionary representation."""
        return {
            "type": "llm",
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LLMScorer:
        """Create scorer from dictionary representation."""
        if data.get("type") != "llm":
            raise ValueError(f"Invalid scorer type: {data.get('type')}")

        return cls(
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"],
        )

    @classmethod
    def get_json_description(cls) -> str:
        """Get the JSON format description for the scorer."""
        return (
            "```json\n"
            "        {\n"
            '            "type": "llm",\n'
            '            "system_prompt": "...",\n'
            '            "user_prompt": "<DESCRIPTION OF THE TASK TO EVALUATE> ... '
            "<INCLUDE ANY CONTEXT WITH VARIABLES USING JINJA2 TEMPLATE STYLE> ... "
            "Respond with JSON in a ```json code block with score between 0 and 1:"
            '\\n```json\\n{\\"reason\\": \\"..\\", \\"score\\": X.XX}\\n```"\n'
            "        }\n"
            "        ```"
        )

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["llm"]},
                "system_prompt": {"type": "string"},
                "user_prompt": {"type": "string"},
            },
            "required": ["type", "system_prompt", "user_prompt"],
        }
