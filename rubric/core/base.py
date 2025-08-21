from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Rubric(Protocol):
    """Rubric protocol."""

    def evaluate(self, include_reason: bool = False, **context: Any) -> tuple[float, str]:
        """Evaluate the rubric."""
        ...

    def reset_scores(self) -> None:
        """Reset the scores of the rubric."""
        ...

    @property
    def score(self) -> float:
        """Get the score from the last evaluation."""
        ...

    @property
    def reason(self) -> str:
        """Get the score reason from the last evaluation."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert the rubric to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        """Create a rubric from a dictionary."""
        ...

    @classmethod
    def generate(cls, task: str, **kwargs: Any) -> Rubric:
        """Generate a rubric for a task."""
        raise NotImplementedError("This method should be implemented by concrete Rubric classes")
