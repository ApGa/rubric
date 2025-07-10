#!/usr/bin/env python3
"""Demo script for the rubric system."""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import rubric
sys.path.insert(0, str(Path(__file__).parent.parent))

from rubric import RubricTree, RubricTreeGenerator
from rubric.examples.basic_examples import create_code_review_rubric, create_simple_rubric


def demo_basic_examples():
    """Demonstrate the basic example rubrics."""
    print("=" * 60)
    print("RUBRIC SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Demo 1: Simple Essay Rubric
    print("\n1. SIMPLE ESSAY RUBRIC")
    print("-" * 30)

    essay_rubric = create_simple_rubric()
    print("Rubric Structure:")
    essay_rubric.print_tree(show_scores=False)

    print("\nTree Statistics:")
    stats = essay_rubric.get_tree_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Demo 2: Code Review Rubric
    print("\n\n2. CODE REVIEW RUBRIC")
    print("-" * 30)

    code_rubric = create_code_review_rubric()
    print("Rubric Structure:")
    code_rubric.print_tree(show_scores=False)

    print("\nTree Statistics:")
    stats = code_rubric.get_tree_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Demo 3: Evaluation with Sample Data
    print("\n\n3. EVALUATION DEMONSTRATION")
    print("-" * 30)

    # Test code evaluation (this should work without LLM)
    code_context = {
        "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"fib({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()
""",
        "test_results": {"passed": 8, "total": 10},
    }

    print("Evaluating code sample...")
    try:
        score = code_rubric.evaluate(code_context)
        print(f"Overall Score: {score:.2f}")
        print("\nDetailed Scores:")
        code_rubric.print_tree(show_scores=True)

        # Get detailed report
        report = code_rubric.get_evaluation_report(code_context)
        print("\nEvaluation Report:")
        print(f"  Overall Score: {report['overall_score']:.2f}")
        print(f"  Nodes Evaluated: {len(report['node_scores'])}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Note: Some scorers may require LLM API access")


def demo_tree_serialization():
    """Demonstrate tree serialization."""
    print("\n\n4. SERIALIZATION DEMONSTRATION")
    print("-" * 30)

    # Create a rubric
    rubric = create_simple_rubric()

    # Save to file
    output_file = Path("sample_rubric.json")
    rubric.save_to_file(output_file)
    print(f"Saved rubric to: {output_file}")

    # Load from file
    loaded_rubric = RubricTree.load_from_file(output_file)
    print(f"Loaded rubric: {loaded_rubric}")

    # Show JSON structure
    print("\nJSON Structure (first 500 chars):")
    with open(output_file) as f:
        content = f.read()
        print(content[:500] + "..." if len(content) > 500 else content)

    # Clean up
    output_file.unlink()
    print(f"Cleaned up: {output_file}")


def demo_tree_generation(task_description: str):
    """Demonstrate LLM-based tree generation."""
    print("\n\n5. LLM TREE GENERATION DEMONSTRATION")
    print("-" * 30)

    try:
        generator = RubricTreeGenerator()
        print(f"Generating rubric for task: {task_description}")

        tree = generator.generate_rubric_tree(
            task_description=task_description, max_depth=2, temperature=0.7
        )

        print("Generated Rubric Structure:")
        tree.print_tree(show_scores=False)

        print("\nTree Statistics:")
        stats = tree.get_tree_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Tree generation failed: {e}")
        print("Note: This requires LLM API access (OPENAI_API_KEY and OPENAI_BASE_URL)")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Rubric System Demo")
    parser.add_argument(
        "--generate-task", type=str, help="Generate a rubric for the specified task using LLM"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip demonstrations that require LLM API access"
    )

    args = parser.parse_args()

    # Always run basic demos
    demo_basic_examples()
    demo_tree_serialization()

    # LLM-based demos
    if not args.skip_llm:
        if args.generate_task:
            demo_tree_generation(args.generate_task)
        else:
            demo_tree_generation("Write a Python function to sort a list of numbers")
    else:
        print("\n\nSkipping LLM demonstrations (use --generate-task to test)")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
