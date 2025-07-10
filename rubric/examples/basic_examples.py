"""Basic example rubric trees for demonstration and testing."""

from ..core import RubricNode, RubricTree
from ..core.scorer import FunctionScorer, LLMScorer, ScriptScorer


def create_simple_rubric() -> RubricTree:
    """Create a simple rubric tree for demonstration.

    Returns:
        A basic rubric tree for evaluating a written essay.
    """
    # Create leaf nodes with scorers
    grammar_node = RubricNode(
        name="Grammar and Spelling",
        description="Evaluate the correctness of grammar and spelling",
        is_critical=True,
        scorer=LLMScorer(
            prompt_template="""
Evaluate the grammar and spelling in the following text:

{{ text }}

Consider:
- Grammatical correctness
- Spelling accuracy  
- Proper punctuation

Return a score between 0 and 1, where:
- 1.0 = Perfect grammar and spelling
- 0.8 = Minor errors that don't impede understanding
- 0.6 = Some errors that occasionally impede understanding
- 0.4 = Frequent errors that often impede understanding
- 0.2 = Many errors that significantly impede understanding
- 0.0 = Errors make the text very difficult to understand

Score: """
        ),
    )

    content_node = RubricNode(
        name="Content Quality",
        description="Evaluate the quality and relevance of content",
        is_critical=True,
        scorer=LLMScorer(
            prompt_template="""
Evaluate the content quality of the following text for the topic "{{ topic }}":

{{ text }}

Consider:
- Relevance to the topic
- Depth of analysis
- Use of examples
- Logical flow of ideas

Return a score between 0 and 1, where:
- 1.0 = Excellent content, highly relevant and insightful
- 0.8 = Good content with minor gaps
- 0.6 = Adequate content but lacks depth
- 0.4 = Poor content with significant issues
- 0.2 = Very poor content, mostly irrelevant
- 0.0 = No relevant content

Score: """
        ),
    )

    style_node = RubricNode(
        name="Writing Style",
        description="Evaluate the writing style and clarity",
        is_critical=False,
        scorer=LLMScorer(
            prompt_template="""
Evaluate the writing style of the following text:

{{ text }}

Consider:
- Clarity of expression
- Sentence variety
- Appropriate tone
- Engaging presentation

Return a score between 0 and 1, where:
- 1.0 = Excellent style, very engaging and clear
- 0.8 = Good style with minor issues
- 0.6 = Adequate style but could be improved
- 0.4 = Poor style that affects readability
- 0.2 = Very poor style, difficult to follow
- 0.0 = Incomprehensible style

Score: """
        ),
    )

    # Create parent node
    root = RubricNode(
        name="Essay Evaluation",
        description="Overall evaluation of essay quality",
        is_critical=False,
        children=[grammar_node, content_node, style_node],
    )

    return RubricTree(
        root=root,
        metadata={
            "description": "Simple essay evaluation rubric",
            "version": "1.0",
            "created_by": "basic_examples.py",
        },
    )


def create_code_review_rubric() -> RubricTree:
    """Create a rubric tree for code review evaluation.

    Returns:
        A rubric tree for evaluating code quality.
    """
    # Functionality nodes
    correctness_node = RubricNode(
        name="Correctness",
        description="Code produces correct results",
        is_critical=True,
        scorer=ScriptScorer(
            script_content="""
import json
import sys

# Read context from stdin
context = json.loads(sys.stdin.read())

code = context.get('code', '')
test_results = context.get('test_results', {})

# Simple scoring based on test results
if not test_results:
    # No tests provided, assume manual review needed
    print(0.5)
else:
    passed = test_results.get('passed', 0)
    total = test_results.get('total', 1)
    score = passed / total if total > 0 else 0
    print(score)
""",
            script_language="python",
        ),
    )

    efficiency_node = RubricNode(
        name="Efficiency",
        description="Code is reasonably efficient",
        is_critical=False,
        scorer=LLMScorer(
            prompt_template="""
Evaluate the efficiency of the following code:

{{ code }}

Consider:
- Time complexity
- Space complexity
- Appropriate algorithm choice
- Unnecessary operations

Return a score between 0 and 1:
- 1.0 = Highly efficient, optimal approach
- 0.8 = Good efficiency with minor improvements possible
- 0.6 = Adequate efficiency but could be better
- 0.4 = Poor efficiency, significant improvements needed
- 0.2 = Very inefficient approach
- 0.0 = Extremely inefficient or incorrect approach

Score: """
        ),
    )

    functionality_node = RubricNode(
        name="Functionality",
        description="Code functionality and correctness",
        is_critical=True,
        children=[correctness_node, efficiency_node],
    )

    # Code quality nodes
    readability_node = RubricNode(
        name="Readability",
        description="Code is easy to read and understand",
        is_critical=False,
        scorer=LLMScorer(
            prompt_template="""
Evaluate the readability of the following code:

{{ code }}

Consider:
- Variable and function naming
- Code organization
- Comments and documentation
- Consistent formatting

Return a score between 0 and 1:
- 1.0 = Excellent readability, self-documenting
- 0.8 = Good readability with minor issues
- 0.6 = Adequate readability but could be clearer
- 0.4 = Poor readability, hard to follow
- 0.2 = Very poor readability
- 0.0 = Unreadable code

Score: """
        ),
    )

    maintainability_node = RubricNode(
        name="Maintainability",
        description="Code is easy to maintain and modify",
        is_critical=False,
        scorer=FunctionScorer(
            function_code="""
def score_function(context):
    code = context.get('code', '')
    
    # Simple heuristics for maintainability
    score = 1.0
    
    # Check for very long functions (rough heuristic)
    lines = code.split('\\n')
    max_function_length = 0
    current_function_length = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith('def ') or line.startswith('class '):
            if current_function_length > max_function_length:
                max_function_length = current_function_length
            current_function_length = 0
        elif line and not line.startswith('#'):
            current_function_length += 1
    
    # Penalize very long functions
    if max_function_length > 50:
        score -= 0.3
    elif max_function_length > 30:
        score -= 0.1
    
    # Check for magic numbers (very basic)
    import re
    magic_numbers = re.findall(r'\\b\\d{2,}\\b', code)
    if len(magic_numbers) > 3:
        score -= 0.2
    
    return max(0.0, score)
""",
            function_name="score_function",
        ),
    )

    quality_node = RubricNode(
        name="Code Quality",
        description="Overall code quality and maintainability",
        is_critical=False,
        children=[readability_node, maintainability_node],
    )

    # Root node
    root = RubricNode(
        name="Code Review",
        description="Comprehensive code review evaluation",
        is_critical=False,
        children=[functionality_node, quality_node],
    )

    return RubricTree(
        root=root,
        metadata={
            "description": "Code review evaluation rubric",
            "version": "1.0",
            "created_by": "basic_examples.py",
        },
    )


def demo_rubric_evaluation() -> None:
    """Demonstrate rubric evaluation with sample data."""
    print("=== Simple Essay Rubric Demo ===")
    essay_rubric = create_simple_rubric()
    essay_rubric.print_tree(show_scores=False)

    # Sample context for essay evaluation
    essay_context = {
        "text": (
            "This is a sample essay about artificial intelligence. AI is transforming "
            "many industries and will continue to have significant impacts on society. "
            "The technology enables automation of complex tasks and provides new "
            "capabilities for data analysis."
        ),
        "topic": "Artificial Intelligence",
    }

    print("\\nEvaluating essay...")
    try:
        score = essay_rubric.evaluate(essay_context)
        print(f"Overall Score: {score:.2f}")
        essay_rubric.print_tree(show_scores=True)
    except Exception as e:
        print(f"Evaluation failed: {e}")

    print("\\n" + "=" * 50)
    print("=== Code Review Rubric Demo ===")
    code_rubric = create_code_review_rubric()
    code_rubric.print_tree(show_scores=False)

    # Sample context for code evaluation
    code_context = {
        "code": """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main() -> None:
    result = fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
""",
        "test_results": {"passed": 8, "total": 10},
    }

    print("\\nEvaluating code...")
    try:
        score = code_rubric.evaluate(code_context)
        print(f"Overall Score: {score:.2f}")
        code_rubric.print_tree(show_scores=True)
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    demo_rubric_evaluation()
