"""Tests for async rubric evaluation paths."""

import asyncio

import pytest

from rubric.core.checklist import RubricChecklistFast
from rubric.core.node import RubricNode
from rubric.core.scorer import LLMScorer
from rubric.core.tree import RubricTree


class FakeAsyncLLMClient:
    """Simple fake client that records async rubric calls."""

    def __init__(self) -> None:
        self.system_calls: list[tuple[str, str, float]] = []
        self.simple_calls: list[tuple[str, float]] = []

    async def asystem_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        **_: object,
    ) -> str:
        self.system_calls.append((system_prompt, user_prompt, temperature))
        if "checklist" in system_prompt.lower():
            return """
```json
{
  "checklist": ["criterion a", "criterion b"],
  "checklist_scores": [1.0, 0.5],
  "reasoning": "Mostly correct with one weaker area.",
  "overall_score": 0.75
}
```
""".strip()

        return """
```json
{
  "reason": "The submitted work satisfies the criterion.",
  "score": 0.8
}
```
""".strip()

    async def asimple_completion(self, prompt: str, temperature: float = 0.7, **_: object) -> str:
        self.simple_calls.append((prompt, temperature))
        return "The parent score reflects strong performance on the only child criterion."


def test_rubric_tree_aevaluate_uses_async_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async tree evaluation should use async LLM scoring and reason generation."""
    fake_client = FakeAsyncLLMClient()

    monkeypatch.setattr(
        "rubric.utils.llm_client.create_llm_client",
        lambda *args, **kwargs: fake_client,
    )

    leaf = RubricNode(
        name="leaf",
        description="Leaf criterion",
        scorer=LLMScorer(
            system_prompt="You are evaluating {subject}.",
            user_prompt='Return JSON for {subject}: {"reason": "...", "score": 0.0}',
        ),
    )
    tree = RubricTree(
        root=RubricNode(
            name="root",
            description="Root criterion",
            children=[leaf],
        )
    )

    score, reason = asyncio.run(tree.aevaluate(include_reason=True, subject="the answer"))

    assert score == 0.8
    assert reason == "The parent score reflects strong performance on the only child criterion."
    assert len(fake_client.system_calls) == 1
    assert len(fake_client.simple_calls) == 1
    assert "the answer" in fake_client.system_calls[0][0]
    assert "the answer" in fake_client.system_calls[0][1]


def test_checklist_aevaluate_uses_async_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async checklist evaluation should use async completion and persist results."""
    fake_client = FakeAsyncLLMClient()

    monkeypatch.setattr(
        "rubric.utils.llm_client.create_llm_client",
        lambda *args, **kwargs: fake_client,
    )

    rubric = RubricChecklistFast(task="Verify the generated summary")

    score, reason = asyncio.run(rubric.aevaluate(include_reason=True, answer="A candidate"))

    assert score == 0.75
    assert reason == "Mostly correct with one weaker area."
    assert rubric.get_checklist() == ["criterion a", "criterion b"]
    assert rubric.get_checklist_scores() == [1.0, 0.5]
    assert len(fake_client.system_calls) == 1
    assert fake_client.simple_calls == []
