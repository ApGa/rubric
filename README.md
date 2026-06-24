# <img src="assets/rubric-icon.png" alt="Rubric Icon" width="100" height="100" style="vertical-align: middle;"> Rubric

Structured verification with LLMs.

## Usage Instructions

Install:

```bash
pip install ai-rubric
```

Configure the LLM used for rubric generation, LLM-based scorers, and parent-node reasoning:

```bash
export RUBRIC_DEFAULT_LLM="openai/gpt-4o-mini"
export OPENAI_API_KEY="..."
# Optional for OpenAI-compatible endpoints:
# export OPENAI_BASE_URL="https://..."
```

### Fast Checklist Evaluation

Use `RubricChecklistFast` when you want a one-call rubric over free-form content:

```python
from rubric.core.checklist import RubricChecklistFast

rubric = RubricChecklistFast(task="Evaluate whether the answer names the capital of France.")
score, reason = rubric.evaluate(
    include_reason=True,
    context="Candidate answer: Paris is the capital of France.",
)

print(score, reason)
```

### Reusable Rubric Trees

Use `RubricTree` when you want explicit, reusable criteria. Scorer functions receive
the context passed to `evaluate()` as globals and return `(reason, score)`.

```python
from rubric import FunctionScorer, RubricNode, RubricTree

tree = RubricTree(
    root=RubricNode(
        name="Capital answer",
        description="Checks whether the answer identifies Paris.",
        children=[
            RubricNode(
                name="Mentions Paris",
                description="The answer should mention Paris.",
                is_critical=True,
                scorer=FunctionScorer(
                    """
def compute_score() -> tuple[str, float]:
    if "paris" in answer.lower():
        return "The answer mentions Paris.", 1.0
    return "The answer does not mention Paris.", 0.0
"""
                ),
            )
        ],
    )
)

score, reason = tree.evaluate(include_reason=True, answer="Paris.")
tree.save_to_file("rubric.json")
```

You can also generate a tree from a task description:

```python
from rubric import RubricTree

tree = RubricTree.generate(
    task="Evaluate whether a summary is accurate and complete.",
    scorer_types=["llm"],
    enforce_structured_output=True,
)
```


## Contributing

1. Install development dependencies: `pip install -e ".[dev]"`
2. Install pre-commit hooks: `pre-commit install`
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request
6. When merging a pull request, please use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
