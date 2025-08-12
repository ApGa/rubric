from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rubric.core.scorer import LLMScorer


@runtime_checkable
class Rubric(Protocol):
    """Rubric protocol."""

    def evaluate(self, include_reason: bool = False, **context: Any) -> tuple[float, str]:
        """Evaluate the rubric."""
        ...

    def reset_scores(self) -> None:
        """Reset the scores of the rubric."""
        ...

    def get_score(self) -> float:
        """Get the score of the rubric."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert the rubric to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        """Create a rubric from a dictionary."""
        ...


class LLMRubric(Rubric):  # TODO: We should make it easier to initialize a default.
    """LLM Rubric."""

    def __init__(self, scorer: LLMScorer):
        """Initialize the LLM Rubric."""
        self.scorer = scorer

    def evaluate(self, include_reason: bool = False, **context: Any) -> tuple[float, str]:
        """Evaluate the rubric."""
        return self.scorer.score(**context)
