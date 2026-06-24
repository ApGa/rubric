"""Regression tests for stale persisted rubric reasons."""

from __future__ import annotations

import asyncio

import pytest

from rubric.core.node import RubricNode
from rubric.core.tree import RubricTree


def _tree_with_stale_reasons() -> RubricTree:
    return RubricTree.from_dict(
        {
            "root": {
                "name": "root",
                "description": "Root criterion",
                "children": [
                    {
                        "name": "leaf",
                        "description": "Leaf criterion",
                        "scorer": {
                            "type": "function",
                            "function_code": (
                                "def compute_score() -> tuple[str, float]:\n"
                                "    return current_reason, current_score\n"
                            ),
                        },
                        "score": 1.0,
                        "reason": "stale leaf reason",
                    }
                ],
                "score": 1.0,
                "reason": "stale parent reason",
            },
            "metadata": {},
        }
    )


def test_evaluate_invalidates_loaded_parent_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-evaluating a loaded tree should not keep a stale parent reason."""

    def fresh_parent_reason(node: RubricNode) -> str:
        return f"fresh parent reason from {node.children[0].reason}"

    monkeypatch.setattr(RubricNode, "_generate_parent_reason", fresh_parent_reason)

    tree = _tree_with_stale_reasons()
    score, reason = tree.evaluate(
        include_reason=True,
        current_reason="fresh leaf reason",
        current_score=0.4,
    )

    assert score == 0.4
    assert reason == "fresh parent reason from fresh leaf reason"
    assert tree.root.to_dict()["reason"] == "fresh parent reason from fresh leaf reason"


def test_aevaluate_invalidates_loaded_parent_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async evaluation should also regenerate stale loaded parent reasons."""

    async def fresh_parent_reason(node: RubricNode) -> str:
        return f"fresh async parent reason from {node.children[0].reason}"

    monkeypatch.setattr(RubricNode, "_agenerate_parent_reason", fresh_parent_reason)

    tree = _tree_with_stale_reasons()
    score, reason = asyncio.run(
        tree.aevaluate(
            include_reason=True,
            current_reason="fresh async leaf reason",
            current_score=0.25,
        )
    )

    assert score == 0.25
    assert reason == "fresh async parent reason from fresh async leaf reason"
    assert tree.root.to_dict()["reason"] == "fresh async parent reason from fresh async leaf reason"
