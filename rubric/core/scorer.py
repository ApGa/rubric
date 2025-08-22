"""Scoring implementations for leaf nodes in the rubric tree."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, cast

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

    # --- Directory-based (human-editable) persistence API ---
    @abstractmethod
    def save_as_dir(self, dir_path: str | Path) -> None:
        """Save this scorer into a directory for easy human editing.

        Implementations should create the directory if missing and write a
        minimal `scorer.json` containing at least the `type`, and any
        additional resources as separate files (e.g., .py or .txt files).
        """
        pass

    @classmethod
    def load_from_dir(cls, dir_path: str | Path) -> "LeafScorer":
        """Load a scorer from a directory produced by `save_as_dir`.

        This method reads `scorer.json` to determine the scorer `type`, then
        delegates to the registered scorer class to construct the instance.
        """
        dir_path = Path(dir_path)
        config_path = dir_path / "scorer.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing scorer.json in {dir_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = json.load(f)

        scorer_type = config.get("type")
        if not scorer_type:
            raise ValueError(f"scorer.json in {dir_path} must include a 'type' field")

        if scorer_type not in SCORER_REGISTRY:
            raise ValueError(f"Unsupported scorer type in {dir_path}: {scorer_type}")

        scorer_cls = SCORER_REGISTRY[scorer_type]
        # Prefer a directory-based loader if provided; otherwise fall back to from_dict
        if hasattr(scorer_cls, "_load_from_dir"):
            loader = getattr(scorer_cls, "_load_from_dir")
            return cast(LeafScorer, loader(dir_path, config))
        return scorer_cls.from_dict(config)


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

    # Directory-based persistence
    def save_as_dir(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save function code in a separate python file for easy editing
        code_path = dir_path / "function.py"
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(self.function_code.rstrip() + "\n")

        # Minimal config that points to the code file
        config = {"type": "function", "function_code_file": code_path.name}
        with open(dir_path / "scorer.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def _load_from_dir(cls, dir_path: str | Path, config: Dict[str, Any]) -> "FunctionScorer":
        dir_p = Path(dir_path)
        code_file = config.get("function_code_file")
        if code_file:
            code_path = dir_p / code_file
            if not code_path.exists():
                raise FileNotFoundError(f"Function code file not found: {code_path}")
            with open(code_path, "r", encoding="utf-8") as f:
                code = f.read()
            return cls(function_code=code)
        # Fallback to inline field for backward compatibility
        if "function_code" in config:
            return cls(function_code=config["function_code"])
        raise ValueError(
            f"Invalid function scorer config in {dir_path}: expected "
            "'function_code_file' or 'function_code'"
        )


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

    # Directory-based persistence
    def save_as_dir(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        sys_path = dir_path / "system_prompt.txt"
        usr_path = dir_path / "user_prompt.txt"
        with open(sys_path, "w", encoding="utf-8") as f:
            f.write(self.system_prompt.rstrip() + "\n")
        with open(usr_path, "w", encoding="utf-8") as f:
            f.write(self.user_prompt.rstrip() + "\n")

        config = {
            "type": "llm",
            "system_prompt_file": sys_path.name,
            "user_prompt_file": usr_path.name,
        }
        with open(dir_path / "scorer.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def _load_from_dir(cls, dir_path: str | Path, config: Dict[str, Any]) -> "LLMScorer":
        dir_p = Path(dir_path)
        sys_file = config.get("system_prompt_file")
        usr_file = config.get("user_prompt_file")
        if sys_file and usr_file:
            sys_path = dir_p / sys_file
            usr_path = dir_p / usr_file
            if not sys_path.exists() or not usr_path.exists():
                raise FileNotFoundError(
                    f"Missing LLM prompt files in {dir_path}: {sys_file}, {usr_file}"
                )
            with open(sys_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            with open(usr_path, "r", encoding="utf-8") as f:
                user_prompt = f.read()
            return cls(system_prompt=system_prompt, user_prompt=user_prompt)

        # Fallback to inline config
        if "system_prompt" in config and "user_prompt" in config:
            return cls(system_prompt=config["system_prompt"], user_prompt=config["user_prompt"])

        raise ValueError(
            f"Invalid LLM scorer config in {dir_path}: expected prompt files or inline prompts"
        )
