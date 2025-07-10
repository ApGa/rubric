"""Scoring implementations for leaf nodes in the rubric tree."""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class LeafScorer(ABC):
    """Abstract base class for leaf node scorers."""

    @abstractmethod
    def score(self, context: Dict[str, Any]) -> float:
        """Compute score for the leaf node.

        Args:
            context: Context data for scoring.

        Returns:
            Score between 0 and 1.
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
        pass


@dataclass
class ScriptScorer(LeafScorer):
    """Scorer that executes a script to compute the score.

    The script should accept context data via stdin (JSON) and output a score
    between 0 and 1 to stdout.
    """

    script_content: str
    script_language: str = "python"
    timeout: int = 30

    def score(self, context: Dict[str, Any]) -> float:
        """Execute the script to compute the score.

        Args:
            context: Context data passed to the script as JSON via stdin.

        Returns:
            Score between 0 and 1.

        Raises:
            ValueError: If script execution fails or returns invalid score.
        """
        import json

        # Create temporary script file
        script_extension = self._get_script_extension()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=script_extension, delete=False
        ) as script_file:
            script_file.write(self.script_content)
            script_path = script_file.name

        try:
            # Prepare command based on language
            cmd = self._get_execution_command(script_path)

            # Execute script with context as stdin
            context_json = json.dumps(context)
            result = subprocess.run(
                cmd, input=context_json, capture_output=True, text=True, timeout=self.timeout
            )

            if result.returncode != 0:
                raise ValueError(f"Script execution failed: {result.stderr}")

            # Parse score from stdout
            try:
                score = float(result.stdout.strip())
                if not (0 <= score <= 1):
                    raise ValueError(f"Score must be between 0 and 1, got {score}")
                return score
            except ValueError as e:
                raise ValueError(f"Invalid score output: {result.stdout.strip()}") from e

        finally:
            # Clean up temporary file
            Path(script_path).unlink(missing_ok=True)

    def _get_script_extension(self) -> str:
        """Get file extension for the script language."""
        extensions = {
            "python": ".py",
            "bash": ".sh",
            "javascript": ".js",
            "node": ".js",
        }
        return extensions.get(self.script_language.lower(), ".txt")

    def _get_execution_command(self, script_path: str) -> list[str]:
        """Get command to execute the script."""
        commands = {
            "python": ["python3", script_path],
            "bash": ["bash", script_path],
            "javascript": ["node", script_path],
            "node": ["node", script_path],
        }

        cmd = commands.get(self.script_language.lower())
        if not cmd:
            raise ValueError(f"Unsupported script language: {self.script_language}")

        return cmd

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorer to dictionary representation."""
        return {
            "type": "script",
            "script_content": self.script_content,
            "script_language": self.script_language,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScriptScorer:
        """Create scorer from dictionary representation."""
        if data.get("type") != "script":
            raise ValueError(f"Invalid scorer type: {data.get('type')}")

        return cls(
            script_content=data["script_content"],
            script_language=data.get("script_language", "python"),
            timeout=data.get("timeout", 30),
        )


@dataclass
class LLMScorer(LeafScorer):
    """Scorer that uses an LLM to compute the score.

    Uses a prompt template to ask the LLM to evaluate the criterion and return a score.
    """

    prompt_template: str
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    def score(self, context: Dict[str, Any]) -> float:
        """Use LLM to compute the score.

        Args:
            context: Context for rendering the prompt.

        Returns:
            Score between 0 and 1.
        """
        from jinja2 import Template

        from ..utils.llm_client import create_llm_client

        # Render prompt template with context
        template = Template(self.prompt_template)
        prompt = template.render(**context)

        # Make LLM call
        client_kwargs = {}
        if self.model:
            client_kwargs["model"] = self.model
        client = create_llm_client(**client_kwargs)

        try:
            response = client.simple_completion(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract score from response
            score = self._extract_score_from_response(response)
            if not (0 <= score <= 1):
                raise ValueError(f"Score must be between 0 and 1, got {score}")

            return score

        except Exception as e:
            raise ValueError(f"LLM scoring failed: {str(e)}") from e

    def _extract_score_from_response(self, response: str) -> float:
        """Extract numerical score from LLM response.

        Args:
            response: LLM response text.

        Returns:
            Extracted score.

        Raises:
            ValueError: If no valid score found in response.
        """
        import re

        # Try to find a decimal number between 0 and 1
        patterns = [
            r"\b(0\.\d+|1\.0+|0|1)\b",  # Decimal between 0 and 1
            r"score[:\s]*([0-9]*\.?[0-9]+)",  # "score: 0.8"
            r"([0-9]*\.?[0-9]+)/10",  # "8/10" format
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    score_str = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    score = float(score_str)

                    # Handle /10 format
                    if "/10" in response and score > 1:
                        score = score / 10

                    if 0 <= score <= 1:
                        return score
                except ValueError:
                    continue

        raise ValueError(f"Could not extract valid score from response: {response}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorer to dictionary representation."""
        return {
            "type": "llm",
            "prompt_template": self.prompt_template,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LLMScorer:
        """Create scorer from dictionary representation."""
        if data.get("type") != "llm":
            raise ValueError(f"Invalid scorer type: {data.get('type')}")

        return cls(
            prompt_template=data["prompt_template"],
            model=data.get("model"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens"),
        )


@dataclass
class FunctionScorer(LeafScorer):
    """Scorer that uses a Python function to compute the score.

    The function should accept context data and return a score between 0 and 1.
    """

    function_code: str
    function_name: str = "score_function"

    def score(self, context: Dict[str, Any]) -> float:
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
            exec(self.function_code, namespace)

            # Get the function
            if self.function_name not in namespace:
                raise ValueError(f"Function '{self.function_name}' not found in code")

            score_func = namespace[self.function_name]

            # Call the function
            score = score_func(context)

            if not isinstance(score, (int, float)):
                raise ValueError(f"Function must return a number, got {type(score)}")

            score = float(score)
            if not (0 <= score <= 1):
                raise ValueError(f"Score must be between 0 and 1, got {score}")

            return score

        except Exception as e:
            raise ValueError(f"Function scoring failed: {str(e)}") from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorer to dictionary representation."""
        return {
            "type": "function",
            "function_code": self.function_code,
            "function_name": self.function_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FunctionScorer:
        """Create scorer from dictionary representation."""
        if data.get("type") != "function":
            raise ValueError(f"Invalid scorer type: {data.get('type')}")

        return cls(
            function_code=data["function_code"],
            function_name=data.get("function_name", "score_function"),
        )


# Factory function to create scorers from dict
def create_scorer_from_dict(data: Dict[str, Any]) -> LeafScorer:
    """Create a scorer from dictionary representation.

    Args:
        data: Dictionary containing scorer configuration.

    Returns:
        LeafScorer instance.

    Raises:
        ValueError: If scorer type is not supported.
    """
    scorer_type = data.get("type")

    if scorer_type == "script":
        return ScriptScorer.from_dict(data)
    elif scorer_type == "llm":
        return LLMScorer.from_dict(data)
    elif scorer_type == "function":
        return FunctionScorer.from_dict(data)
    else:
        raise ValueError(f"Unsupported scorer type: {scorer_type}")


# Update LeafScorer to use the factory
setattr(LeafScorer, "from_dict", staticmethod(create_scorer_from_dict))
