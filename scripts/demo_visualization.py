#!/usr/bin/env python3
"""Demo script showcasing interactive Plotly tree visualization features."""

import sys
from pathlib import Path

# Add the rubric package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rubric.core import RubricNode, RubricTree
from rubric.core.scorer import LLMScorer


def create_sample_tree() -> RubricTree:
    """Create a sample rubric tree for demonstration."""

    # Create leaf nodes with scorers
    code_quality = RubricNode(
        name="Code Quality",
        description=(
            "Assess the overall quality of the code including readability, "
            "structure, and best practices"
        ),
        is_critical=True,
        scorer=LLMScorer(
            prompt_template=(
                "Evaluate the code quality based on readability, structure, and "
                "adherence to best practices. Return a score between 0 and 1."
            )
        ),
    )

    documentation = RubricNode(
        name="Documentation",
        description=("Check if the code is properly documented with clear comments and docstrings"),
        is_critical=False,
        scorer=LLMScorer(
            prompt_template=(
                "Assess the quality and completeness of code documentation. "
                "Return a score between 0 and 1."
            )
        ),
    )

    functionality = RubricNode(
        name="Functionality",
        description="Verify that the code works correctly and meets requirements",
        is_critical=True,
        scorer=LLMScorer(
            prompt_template=(
                "Test and evaluate if the code functions correctly and meets all "
                "requirements. Return a score between 0 and 1."
            )
        ),
    )

    error_handling = RubricNode(
        name="Error Handling",
        description="Check if the code properly handles errors and edge cases",
        is_critical=False,
        scorer=LLMScorer(
            prompt_template=(
                "Evaluate how well the code handles errors and edge cases. "
                "Return a score between 0 and 1."
            )
        ),
    )

    # Create parent nodes
    implementation = RubricNode(
        name="Implementation Quality",
        description="Overall assessment of code implementation",
        is_critical=True,
        children=[code_quality, documentation],
    )

    correctness = RubricNode(
        name="Correctness",
        description="Assessment of functional correctness and robustness",
        is_critical=True,
        children=[functionality, error_handling],
    )

    # Create root node
    root = RubricNode(
        name="Code Review Rubric",
        description="Comprehensive rubric for evaluating code submissions",
        is_critical=False,
        children=[implementation, correctness],
    )

    return RubricTree(root=root)


def demonstrate_text_visualization():
    """Demonstrate text-based tree visualization."""
    print("=" * 60)
    print("TEXT-BASED VISUALIZATION DEMO")
    print("=" * 60)

    tree = create_sample_tree()

    print("\n1. Basic tree structure:")
    tree.print_tree(show_scores=False)

    print("\n2. Enhanced text tree with emojis:")
    text_tree = tree.generate_text_tree(show_scores=False, max_width=80)
    print(text_tree)

    print("\n3. Tree statistics:")
    stats = tree.get_tree_stats()
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


def demonstrate_plotly_visualization():
    """Demonstrate interactive Plotly tree visualization."""
    print("\n" + "=" * 60)
    print("INTERACTIVE PLOTLY VISUALIZATION DEMO")
    print("=" * 60)

    tree = create_sample_tree()

    print("\n1. Creating interactive Plotly tree visualization...")
    print("   This will display in your browser or Jupyter notebook!")

    # Use the simple plot method
    print("\n   Using tree.plot()...")
    fig = tree.plot(show_scores=False, title="Demo: Code Review Rubric Tree")
    if fig:
        print("   ✅ Plotly tree visualization displayed!")
    else:
        print("   ❌ Plotly not available - install with: pip install plotly")
        return

    print("\n2. Trying different layouts...")

    # Hierarchical layout (default)
    print("   • Hierarchical layout")
    tree.plot(layout="hierarchical", show_scores=False, title="Hierarchical Layout")

    # Circular layout
    print("   • Circular layout")
    tree.plot_network(layout="circular", show_scores=False, title="Circular Layout")

    # Spring layout
    print("   • Spring layout")
    tree.plot_network(layout="spring", show_scores=False, title="Spring Layout")

    print("\n📝 Enhanced large-node visualization features:")
    print("   • 📖 Full titles and descriptions visible in nodes")
    print("   • 🔍 Much larger nodes for better readability")
    print("   • 📝 Multi-line text formatting with word wrapping")
    print("   • 🎨 Beautiful modern styling with better colors")
    print("   • ⚙️ Scorer information displayed (LLM vs Script)")
    print("   • 💬 Prompt templates shown in hover details")
    print("   • 📏 Smart node sizing based on content importance")
    print("   • 🎯 Improved spacing to prevent overlap")
    print("   • 🔄 Zoom, pan, and interact with the tree")
    print("   • 🌈 Color-coded scores and criticality")
    print("   • 🔶 Different symbols for different scorer types")


def demonstrate_with_scores():
    """Demonstrate visualization with computed scores."""
    print("\n" + "=" * 60)
    print("PLOTLY VISUALIZATION WITH SCORES DEMO")
    print("=" * 60)

    tree = create_sample_tree()

    # Simulate some scores for demonstration
    # In practice, these would be computed by evaluating the tree
    print("\n1. Simulating score computation...")

    # Manually set some scores for demo purposes
    all_nodes = tree.get_all_nodes()
    for i, node in enumerate(all_nodes):
        if node.is_leaf:
            # Simulate different score levels
            score = 0.9 - (i * 0.15)  # Varying scores
            score = max(0.1, min(1.0, score))  # Keep in valid range
            node._score = score
            print(f"   {node.name}: {score:.2f}")

    print("\n2. Creating Plotly visualization with scores...")
    fig = tree.plot(show_scores=True, title="Demo: Rubric Tree with Scores", width=1400, height=900)

    if fig:
        print("✅ Enhanced Plotly visualization with scores displayed!")
        print("   Notice the improved features:")
        print("   🟢 Green: High scores (≥0.8)")
        print("   🟡 Yellow: Good scores (0.6-0.8)")
        print("   🟠 Orange: Fair scores (0.4-0.6)")
        print("   🔴 Red: Low scores (<0.4)")
        print("   🤖 Circles: LLM scorers")
        print("   🔶 Diamonds: Script scorers")
        print("   📏 Node sizes reflect importance/children count")
        print("   💬 Hover to see scorer details and prompts!")
    else:
        print("❌ Plotly not available")

    print("\n3. Enhanced text tree with scores:")
    text_tree = tree.generate_text_tree(show_scores=True, max_width=80)
    print(text_tree)

    print("\n4. Trying different layouts with scores...")

    # Network layout with scores
    print("   • Network layout with scores")
    tree.plot_network(layout="hierarchical", show_scores=True, title="Network Layout with Scores")


def main():
    """Run all visualization demos."""
    print("🌳 INTERACTIVE RUBRIC TREE VISUALIZATION DEMO")
    print("This demo showcases interactive Plotly visualization for rubric trees.")

    try:
        # Run demos
        demonstrate_text_visualization()
        demonstrate_plotly_visualization()
        demonstrate_with_scores()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\n🎯 Key takeaways:")
        print("   • Use tree.plot() for quick interactive visualizations")
        print("   • Use tree.plot_network() for different layout styles")
        print("   • Hover over nodes to see detailed information")
        print("   • Score-based color coding helps identify issues")
        print("   • Works seamlessly in Jupyter notebooks")
        print("\n💡 Install Plotly if you haven't already:")
        print("   pip install plotly")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
