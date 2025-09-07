"""Generator for creating rubric trees using LLMs."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

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
        enforce_structured_output: bool = False,
        reasoning_effort: str | None = None,
        compute_strategy: Literal["default", "mind2web2"] = "default",
        critical_node_weight: float = 0.7,
    ) -> RubricTree:
        """Generate a rubric tree for evaluating a task.

        Args:
            task: Description of the task to create a rubric for.
            rubric_gen_prompt_context: Additional context for rubric generation.
            temperature: Temperature for LLM generation.
            max_tokens: Maximum number of tokens to generate.
            scorer_types: List of scorer types to allow for leaf nodes.
            reasoning_effort: String for reasoning effort information.
        Returns:
            Generated RubricTree.
        """
        # Prepare context for prompt

        # Generate rubric structure using LLM
        system_prompt = self.prompt_retriever.get_prompt(
            "generate-rubric-tree-system",
            compute_strategy=compute_strategy,
            critical_node_weight=critical_node_weight,
        )
        user_prompt = self.prompt_retriever.get_prompt(
            "generate-rubric-tree-user",
            task=task,
            rubric_gen_prompt_context=rubric_gen_prompt_context,
            rubric_gen_generation_guidelines=rubric_gen_generation_guidelines,
            scorer_types=scorer_types,
            scorer_formats="\n".join(
                SCORER_REGISTRY[scorer_type].get_json_description() for scorer_type in scorer_types
            ),
            compute_strategy=compute_strategy,
            critical_node_weight=critical_node_weight,
        )

        call_kwargs: Dict[str, Any] = {}
        if enforce_structured_output:
            call_kwargs["response_format"] = self._build_rubric_node_response_format_schema(
                allowed_scorer_types=scorer_types
            )

        response = self.llm_client.system_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            **call_kwargs,
        )

        # Parse JSON response
        try:
            parsed = self._extract_json_from_response(response)
            rubric_data = (
                parsed if isinstance(parsed, dict) and "root" in parsed else {"root": parsed}
            )
            tree = RubricTree.from_dict(rubric_data)
            tree.metadata["task"] = task
            tree.metadata["compute_strategy"] = compute_strategy
            if compute_strategy == "default":
                tree.metadata["critical_node_weight"] = critical_node_weight
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

    def _build_rubric_node_response_format_schema(
        self, allowed_scorer_types: list[str]
    ) -> Dict[str, Any]:
        """Build OpenAI-compatible response_format for enforcing rubric node JSON schema.

        This schema enforces that the model returns a single rubric node object which
        can either contain children (recursive) or a scorer, but not both.
        """
        # Build scorer variants based on allowed types, using schemas provided by the scorers
        scorer_variants: list[Dict[str, Any]] = []
        for scorer_type in allowed_scorer_types:
            scorer_cls = SCORER_REGISTRY.get(scorer_type)
            if scorer_cls is None:
                continue
            try:
                scorer_schema = scorer_cls.get_json_schema()
                scorer_variants.append(scorer_schema)
            except Exception:
                # If a scorer doesn't provide a valid schema, skip it
                continue

        # Fallback to permissive scorer if registry is empty
        if not scorer_variants:
            scorer_variants.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"type": {"type": "string"}},
                    "required": ["type"],
                }
            )

        # Rely on scorer-provided schemas. Each scorer's get_json_schema() must be a strict object
        # schema with additionalProperties: false and include all properties in required.

        # NOTE: OpenAI/LiteLLM requires the top-level schema to explicitly be of type "object".
        # We define the node schema at the top-level and use a self-reference ("$ref": "#")
        # for recursion in the children items. This avoids a top-level $ref which some
        # providers reject and ensures "type" is not None.
        schema: Dict[str, Any] = {
            "type": "object",
            "required": ["root"],
            "properties": {"root": {"$ref": "#/definitions/node"}},
            "additionalProperties": False,
            "definitions": {
                "scorer": {"anyOf": scorer_variants},
                "node": {
                    "type": "object",
                    "required": ["name", "description"],
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "is_critical": {"type": "boolean"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/node"},
                            "minItems": 1,
                        },
                        "scorer": {"$ref": "#/definitions/scorer"},
                    },
                    "anyOf": [
                        {"type": "object", "additionalProperties": False, "required": ["children"]},
                        {"type": "object", "additionalProperties": False, "required": ["scorer"]},
                    ],
                    "additionalProperties": False,
                },
            },
        }

        # OpenAI-style response_format wrapper
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "RubricNode",
                "schema": schema,
                "strict": True,
            },
        }

        return response_format
