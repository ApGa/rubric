"""Generator for creating rubric trees using LLMs."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict

from ..core import RubricTree
from ..core.scorer import SCORER_REGISTRY
from ..utils.llm_client import LLMClient, create_llm_client
from ..utils.prompt_retriever import PromptRetriever


@dataclass
class RubricTreeGenerator:
    """Generator for creating rubric trees using LLMs."""

    llm_client: LLMClient = field(default_factory=create_llm_client)
    prompt_retriever: PromptRetriever = field(default_factory=PromptRetriever)

    def generate_rubric_tree(
        self,
        task: str,
        rubric_gen_prompt_context: str = "",
        rubric_gen_generation_guidelines: str = "",
        temperature: float = 0.7,
        max_tokens: int = 10000,
        scorer_types: list[str] = list(SCORER_REGISTRY.keys()),
    ) -> RubricTree:
        """Generate a rubric tree for evaluating a task.

        Args:
            task: Description of the task to create a rubric for.
            rubric_gen_prompt_context: Additional context for rubric generation.
            temperature: Temperature for LLM generation.
            max_tokens: Maximum number of tokens to generate.
            scorer_types: List of scorer types to allow for leaf nodes.
        Returns:
            Generated RubricTree.
        """
        # Prepare context for prompt


        # Generate rubric structure using LLM
        system_prompt = self.prompt_retriever.get_prompt("generate-rubric-tree-system")
        user_prompt = self.prompt_retriever.get_prompt("generate-rubric-tree-user", 
            task=task, 
            rubric_gen_prompt_context=rubric_gen_prompt_context, 
            rubric_gen_generation_guidelines=rubric_gen_generation_guidelines,
            scorer_types=scorer_types,
            scorer_formats='\n'.join(
                SCORER_REGISTRY[scorer_type].get_json_description() for scorer_type in scorer_types
            )
        )

        response = self.llm_client.system_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print(response)

        # Parse JSON response
        try:
            rubric_data = {
                "root": self._extract_json_from_response(response),
            }
            tree = RubricTree.from_dict(rubric_data)
            tree.metadata["task"] = task
            return tree
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
