from __future__ import annotations

import json
import re
from typing import Any

from ..utils.llm_client import create_llm_client
from ..utils.llm_tools import LLM_MODEL_NAME
from ..utils.prompt_retriever import PromptRetriever
from .base import Rubric


class RubricChecklistFast(Rubric):
    """A rubric that prompts an LLM to reason about a tasks correctness
    by generating and evaluating against a checklist, all within a single call.
    """

    def __init__(self, task: str):
        self.task = task
        self._last_score: float | None = None
        self._last_reason: str = ""
        self._last_checklist: list[str] = []
        self._last_checklist_scores: list[float] = []
        self._last_parsed_response: dict[str, Any] | None = None
        self.prompt_retriever = PromptRetriever()

    def evaluate(
        self, include_reason: bool = False, temperature: float = 1.0, **context: Any
    ) -> tuple[float, str]:
        """Evaluate the task using an LLM-generated checklist in a single call.

        Args:
            include_reason: Whether to include reasoning in the response.
            **context: Additional context variables for evaluation.

        Returns:
            Tuple of (score, reason) where score is between 0 and 1.
        """
        try:
            # Get the system and user prompts for checklist evaluation
            system_prompt = self.prompt_retriever.get_prompt(
                "generate-rubric-checklist-fast-system"
            )

            # Prepare context for user prompt
            prompt_context = {"task": self.task}
            prompt_context.update(context)

            user_prompt = self.prompt_retriever.get_prompt(
                "generate-rubric-checklist-fast-user", **prompt_context
            )

            # Create LLM client and make request
            llm_client = create_llm_client(model=LLM_MODEL_NAME)
            response = llm_client.system_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )

            # Parse the JSON response
            parsed_response = self._parse_response(response)

            # Extract overall score and reasoning
            overall_score = parsed_response.get("overall_score", 0.0)
            reasoning = parsed_response.get("reasoning", "No reasoning provided")
            checklist = parsed_response.get("checklist", [])
            checklist_scores = parsed_response.get("checklist_scores", [])

            # Store the results
            self._last_score = float(overall_score)
            self._last_reason = reasoning
            self._last_checklist = list(checklist) if checklist else []
            self._last_checklist_scores = (
                [float(score) for score in checklist_scores] if checklist_scores else []
            )
            self._last_parsed_response = parsed_response

            # Validate score range
            if not (0 <= self._last_score <= 1):
                raise ValueError(f"Score must be between 0 and 1, got {self._last_score}")

            if include_reason:
                return self._last_score, self._last_reason
            else:
                return self._last_score, ""

        except Exception as e:
            raise ValueError(f"Checklist evaluation failed: {str(e)}") from e

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response to extract structured data.

        Args:
            response: Raw response from the LLM.

        Returns:
            Parsed response as dictionary.
        """
        try:
            # Try to find JSON code block first
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)

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

            # Validate required fields
            if not isinstance(parsed_response, dict):
                raise ValueError("Response must be a JSON object")

            required_fields = ["checklist", "checklist_scores", "reasoning", "overall_score"]
            for field in required_fields:
                if field not in parsed_response:
                    raise ValueError(f"Missing required field: {field}")

            return parsed_response

        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON response: {str(e)}") from e

    def reset_scores(self) -> None:
        """Reset the scores of the rubric."""
        self._last_score = None
        self._last_reason = ""
        self._last_checklist = []
        self._last_checklist_scores = []
        self._last_parsed_response = None

    def get_score(self) -> float:
        """Get the score of the rubric.

        Returns:
            The last computed score, or 0.0 if no evaluation has been performed.
        """
        return self._last_score if self._last_score is not None else 0.0

    def get_checklist(self) -> list[str]:
        """Get the last generated checklist.

        Returns:
            List of checklist items from the last evaluation, or empty list if none.
        """
        return self._last_checklist.copy()

    def get_checklist_scores(self) -> list[float]:
        """Get the last checklist scores.

        Returns:
            List of scores for each checklist item from the last evaluation, or empty list if none.
        """
        return self._last_checklist_scores.copy()

    def get_reasoning(self) -> str:
        """Get the reasoning from the last evaluation.

        Returns:
            The reasoning string from the last evaluation, or empty string if none.
        """
        return self._last_reason

    def get_full_evaluation(self) -> dict[str, Any] | None:
        """Get the complete parsed response from the last evaluation.

        Returns:
            The full parsed response dictionary, or None if no evaluation has been performed.
        """
        return self._last_parsed_response.copy() if self._last_parsed_response else None

    def to_dict(self) -> dict[str, Any]:
        """Convert the rubric to a dictionary."""
        result = {
            "type": "checklist_fast",
            "task": self.task,
        }

        if self._last_score is not None:
            result["last_score"] = self._last_score
            result["last_reason"] = self._last_reason
            result["last_checklist"] = self._last_checklist
            result["last_checklist_scores"] = self._last_checklist_scores

            # Include the full parsed response for complete evaluation details
            if self._last_parsed_response is not None:
                result["last_parsed_response"] = self._last_parsed_response

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricChecklistFast:
        """Create a rubric from a dictionary."""
        if data.get("type") != "checklist_fast":
            raise ValueError(f"Invalid rubric type: {data.get('type')}")

        instance = cls(task=data["task"])

        # Restore previous evaluation results if available
        if "last_score" in data:
            instance._last_score = data["last_score"]
            instance._last_reason = data.get("last_reason", "")
            instance._last_checklist = data.get("last_checklist", [])
            instance._last_checklist_scores = data.get("last_checklist_scores", [])
            instance._last_parsed_response = data.get("last_parsed_response")

        return instance
