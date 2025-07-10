"""Generator for creating rubric trees using LLMs."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..core import RubricNode, RubricTree
from ..core.scorer import LLMScorer, ScriptScorer
from ..utils.llm_client import LLMClient, create_llm_client
from ..utils.prompt_retriever import PromptRetriever


@dataclass
class RubricTreeGenerator:
    """Generator for creating rubric trees using LLMs."""

    llm_client: LLMClient = field(default_factory=create_llm_client)
    prompt_retriever: PromptRetriever = field(default_factory=PromptRetriever)

    def generate_rubric_tree(
        self,
        task_description: str,
        additional_context: Optional[Dict[str, Any]] = None,
        max_depth: int = 3,
        temperature: float = 0.7,
    ) -> RubricTree:
        """Generate a rubric tree for evaluating a task.

        Args:
            task_description: Description of the task to create a rubric for.
            additional_context: Additional context for rubric generation.
            max_depth: Maximum depth of the generated tree.
            temperature: Temperature for LLM generation.

        Returns:
            Generated RubricTree.
        """
        # Prepare context for prompt
        context = {
            "task_description": task_description,
            "max_depth": max_depth,
            **(additional_context or {}),
        }

        # Generate rubric structure using LLM
        prompt = self.prompt_retriever.get_prompt("generate-tree-rubric-system", **context)

        # Add specific instructions for JSON output
        full_prompt = f"""{prompt}

Please generate a comprehensive rubric tree for the following task:

Task: {task_description}

Return the rubric as a JSON structure with the following format:
{{
    "name": "Root criterion name",
    "description": "Detailed description of what this criterion evaluates",
    "is_critical": true/false,
    "children": [
        {{
            "name": "Child criterion name",
            "description": "Description",
            "is_critical": true/false,
            "children": [...] // or "scorer" for leaf nodes
        }}
    ]
}}

For leaf nodes, instead of "children", include:
{{
    "scorer": {{
        "type": "llm",
        "prompt_template": "Evaluate whether... Return a score between 0 and 1."
    }}
}}

Make sure the rubric is comprehensive, follows the scoring rules described above, 
and has appropriate critical/non-critical designations."""

        response = self.llm_client.simple_completion(
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=4000,
        )

        # Parse JSON response
        try:
            rubric_data = self._extract_json_from_response(response)
            root_node = self._create_node_from_data(rubric_data)
            return RubricTree(root=root_node)
        except Exception as e:
            raise ValueError(f"Failed to generate rubric tree: {str(e)}") from e

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        import re

        # Look for JSON blocks
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            json_str = matches[0]
        else:
            # Try to find JSON without code blocks
            json_start = response.find("{")
            json_end = response.rfind("}")
            if json_start != -1 and json_end != -1:
                json_str = response[json_start : json_end + 1]
            else:
                raise ValueError("No JSON found in response")

        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
            else:
                raise ValueError("JSON response is not a dictionary")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")

    def _create_node_from_data(self, data: Dict[str, Any]) -> RubricNode:
        """Create a RubricNode from dictionary data."""
        if "scorer" in data:
            # Leaf node
            scorer_data = data["scorer"]
            scorer = self._create_scorer_from_data(scorer_data)
            node = RubricNode(
                name=data["name"],
                description=data["description"],
                is_critical=data.get("is_critical", False),
                scorer=scorer,
            )
        elif "children" in data:
            # Parent node - create children first
            children = [self._create_node_from_data(child_data) for child_data in data["children"]]
            node = RubricNode(
                name=data["name"],
                description=data["description"],
                is_critical=data.get("is_critical", False),
                children=children,
            )
        else:
            raise ValueError(f"Node '{data['name']}' must have either children or scorer")

        return node

    def _create_scorer_from_data(self, data: Dict[str, Any]) -> Any:
        """Create a scorer from dictionary data."""
        scorer_type = data.get("type", "llm")

        if scorer_type == "llm":
            return LLMScorer(
                prompt_template=data["prompt_template"],
                model=data.get("model"),
                temperature=data.get("temperature", 0.0),
                max_tokens=data.get("max_tokens"),
            )
        elif scorer_type == "script":
            return ScriptScorer(
                script_content=data["script_content"],
                script_language=data.get("script_language", "python"),
                timeout=data.get("timeout", 30),
            )
        else:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")

    def refine_rubric_tree(
        self,
        tree: RubricTree,
        feedback: str,
        temperature: float = 0.7,
    ) -> RubricTree:
        """Refine an existing rubric tree based on feedback.

        Args:
            tree: Existing rubric tree to refine.
            feedback: Feedback on how to improve the rubric.
            temperature: Temperature for LLM generation.

        Returns:
            Refined RubricTree.
        """
        # Convert tree to dict for context
        tree_dict = tree.to_dict()

        prompt = f"""Here is an existing rubric tree:

{json.dumps(tree_dict, indent=2)}

Please refine this rubric based on the following feedback:

{feedback}

Return the improved rubric as a JSON structure with the same format as above.
Make sure to maintain the overall structure while addressing the feedback."""

        response = self.llm_client.simple_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=4000,
        )

        try:
            rubric_data = self._extract_json_from_response(response)
            root_node = self._create_node_from_data(rubric_data)
            return RubricTree(root=root_node)
        except Exception as e:
            raise ValueError(f"Failed to refine rubric tree: {str(e)}") from e

    def generate_scorer_for_criterion(
        self,
        criterion_name: str,
        criterion_description: str,
        scorer_type: str = "llm",
        temperature: float = 0.3,
    ) -> Any:
        """Generate a scorer for a specific criterion.

        Args:
            criterion_name: Name of the criterion.
            criterion_description: Description of what to evaluate.
            scorer_type: Type of scorer to generate ("llm" or "script").
            temperature: Temperature for LLM generation.

        Returns:
            Generated scorer instance.
        """
        if scorer_type == "llm":
            prompt = f"""Create a prompt template for evaluating the following criterion:

Criterion: {criterion_name}
Description: {criterion_description}

The prompt template should:
1. Clearly explain what to evaluate
2. Provide specific guidance on scoring
3. Ask for a score between 0 and 1
4. Use Jinja2 template syntax to access context variables

Return only the prompt template text."""

            response = self.llm_client.simple_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=1000,
            )

            return LLMScorer(prompt_template=response.strip())

        elif scorer_type == "script":
            prompt = f"""Create a Python script for evaluating the following criterion:

Criterion: {criterion_name}
Description: {criterion_description}

The script should:
1. Read JSON context data from stdin
2. Evaluate the criterion based on the context
3. Print a score between 0 and 1 to stdout
4. Handle errors gracefully

Return only the Python code."""

            response = self.llm_client.simple_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=1000,
            )

            return ScriptScorer(script_content=response.strip())

        else:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")

    def validate_and_fix_tree(self, tree: RubricTree) -> RubricTree:
        """Validate a rubric tree and attempt to fix any issues.

        Args:
            tree: Tree to validate and fix.

        Returns:
            Fixed tree (or original if no issues found).
        """
        errors = tree.validate_tree()
        if not errors:
            return tree

        # Try to fix common issues
        for node in tree.get_all_nodes():
            # Fix leaf nodes without scorers
            if node.is_leaf and not node.scorer:
                scorer = self.generate_scorer_for_criterion(node.name, node.description, "llm")
                node.set_scorer(scorer)

        # Validate again
        remaining_errors = tree.validate_tree()
        if remaining_errors:
            raise ValueError(f"Could not fix all tree issues: {remaining_errors}")

        return tree

    def generate_and_plot(
        self,
        task_description: str,
        additional_context: Optional[Dict[str, Any]] = None,
        max_depth: int = 3,
        temperature: float = 0.7,
        layout: str = "hierarchical",
        show_scores: bool = False,
    ) -> RubricTree:
        """Generate a rubric tree and display it interactively with Plotly.

        Args:
            task_description: Description of the task to create a rubric for.
            additional_context: Additional context for rubric generation.
            max_depth: Maximum depth of the generated tree.
            temperature: Temperature for LLM generation.
            layout: Layout algorithm for visualization.
            show_scores: Whether to show scores (False by default since tree is new).

        Returns:
            Tuple of (generated_tree, plotly_figure).
        """
        # Generate the tree
        tree = self.generate_rubric_tree(
            task_description=task_description,
            additional_context=additional_context,
            max_depth=max_depth,
            temperature=temperature,
        )

        # Create interactive plot
        tree.plot(
            show_scores=show_scores,
            layout=layout,
            title=f"Generated Rubric: {task_description[:50]}...",
            width=1400,
            height=900,
        )

        return tree
