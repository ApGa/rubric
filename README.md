# Rubric AI

A comprehensive tree-based rubric system for structured evaluation and verification using LLMs.

## Overview

Rubric AI provides a flexible framework for creating and evaluating hierarchical rubrics. The system is designed around a tree structure where:

- **Nodes** represent evaluation criteria
- **Leaf nodes** use scorers (LLM, script, or function-based) to compute scores
- **Parent nodes** aggregate child scores using configurable rules
- **Critical vs non-critical** designations affect scoring behavior

## Key Features

- 🌳 **Tree-based rubric structure** with hierarchical criteria
- 🤖 **Multiple scorer types**: LLM-based, script-based, and function-based
- ⚖️ **Flexible scoring rules** with critical/non-critical designations
- 🔄 **Serialization support** for saving/loading rubrics
- 🎯 **LLM-powered generation** of rubric trees
- 📊 **Comprehensive evaluation reports**
- 🧪 **Built-in validation** and error checking

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Creating a Simple Rubric

```python
from rubric import RubricNode, RubricTree
from rubric.core.scorer import LLMScorer

# Create leaf nodes with scorers
grammar_node = RubricNode(
    name="Grammar",
    description="Evaluate grammar and spelling",
    is_critical=True,
    scorer=LLMScorer(
        prompt_template="Evaluate the grammar in: {{ text }}. Return score 0-1."
    )
)

content_node = RubricNode(
    name="Content",
    description="Evaluate content quality", 
    is_critical=True,
    scorer=LLMScorer(
        prompt_template="Evaluate content quality in: {{ text }}. Return score 0-1."
    )
)

# Create parent node
root = RubricNode(
    name="Essay Evaluation",
    description="Overall essay quality",
    children=[grammar_node, content_node]
)

# Create tree and evaluate
tree = RubricTree(root=root)
score = tree.evaluate({"text": "Your essay text here..."})
print(f"Overall score: {score}")
```

### Using Pre-built Examples

```python
from rubric.examples import create_simple_rubric, create_code_review_rubric

# Use pre-built rubrics
essay_rubric = create_simple_rubric()
code_rubric = create_code_review_rubric()

# Evaluate
essay_score = essay_rubric.evaluate({
    "text": "Essay content...",
    "topic": "AI Ethics"
})

code_score = code_rubric.evaluate({
    "code": "def hello(): print('Hello!')",
    "test_results": {"passed": 10, "total": 10}
})
```

### Generating Rubrics with LLM

```python
from rubric import RubricTreeGenerator

generator = RubricTreeGenerator()
tree = generator.generate_rubric_tree(
    task_description="Evaluate a Python function for correctness and style",
    max_depth=3
)

tree.print_tree()
```

## Scoring Rules

The rubric system implements specific scoring rules based on critical/non-critical designations:

1. **Parent node score is 0** if any critical child has score 0
2. **Parent node score is average of non-critical children** if all critical children have score 1
3. **Parent node score is average of all children** if no critical children exist
4. **Leaf node scores** are computed by their assigned scorers

## Scorer Types

### LLM Scorer
Uses language models to evaluate criteria:

```python
from rubric.core.scorer import LLMScorer

scorer = LLMScorer(
    prompt_template="Evaluate {{ criterion }} in: {{ text }}. Score 0-1: ",
    temperature=0.0
)
```

### Script Scorer
Executes external scripts for evaluation:

```python
from rubric.core.scorer import ScriptScorer

scorer = ScriptScorer(
    script_content="""
import json, sys
context = json.loads(sys.stdin.read())
# Your scoring logic here
print(0.85)  # Output score
""",
    script_language="python"
)
```

### Function Scorer
Uses Python functions for evaluation:

```python
from rubric.core.scorer import FunctionScorer

scorer = FunctionScorer(
    function_code="""
def score_function(context):
    # Your scoring logic here
    return 0.75
""",
    function_name="score_function"
)
```

## Environment Setup

For LLM-based features, set these environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-base-url"
```

## Demo

Run the demonstration script:

```bash
python scripts/demo_rubric.py --skip-llm  # Without LLM features
python scripts/demo_rubric.py --generate-task "Evaluate code quality"  # With LLM
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Project Structure

```
rubric/
├── core/                 # Core rubric system
│   ├── node.py          # RubricNode implementation
│   ├── tree.py          # RubricTree implementation
│   └── scorer.py        # Scorer implementations
├── generate/            # LLM-based generation
│   └── tree_generator.py
├── examples/            # Example rubrics
│   └── basic_examples.py
├── utils/               # Utility modules
│   ├── llm_client.py    # LLM client
│   └── prompt_retriever.py  # Prompt management
└── prompts/             # Jinja2 templates
    └── generate-tree-rubric-system.jinja
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
