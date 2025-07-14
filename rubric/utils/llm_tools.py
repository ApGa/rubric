"""LLM functions that are suitable to include as tools to expose to LLMs"""

import inspect
import os
from typing import Callable, List, Union

from rubric.constants import RUBRIC_DEFAULT_LLM
from rubric.utils.llm_client import create_llm_client

LLM_MODEL_NAME = os.environ.get("RUBRIC_LLM_TOOL_MODEL_NAME", RUBRIC_DEFAULT_LLM)
VLM_MODEL_NAME = os.environ.get("RUBRIC_VLM_TOOL_MODEL_NAME", LLM_MODEL_NAME)

llm_client = create_llm_client(model=LLM_MODEL_NAME)
vlm_client = create_llm_client(model=VLM_MODEL_NAME)


def llm_call(prompt: str, temperature: float = 0.7, max_tokens: int | None = None) -> str:
    """Call the LLM client with the given prompt.

    Args:
        prompt: The prompt to send to the LLM.
        temperature: The temperature to use for the LLM.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        The response from the LLM.
    """
    return llm_client.simple_completion(prompt, temperature, max_tokens)


def vlm_call(
    prompt: str,
    images: List[Union[str, bytes]],
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Call the Vision LM client with the given prompt and images.

    Args:
        prompt: The prompt to send to the VLM.
        images: The images to send to the VLM. Each image can be:
            - File path (string) - will be read and base64 encoded
            - Base64 encoded string
            - Raw bytes - will be base64 encoded
        temperature: The temperature to use for the VLM.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        The response from the VLM.
    """
    return vlm_client.vision_completion(prompt, images, temperature, max_tokens)


def generate_prompt_descriptions_for_functions(functions: List[Callable]) -> List[str]:
    """Generate prompt descriptions for the given functions.

    Args:
        functions: The functions to generate prompt descriptions for.

    Returns:
        The prompt descriptions for the given functions.
    """
    # Return a list of strings with the function signature and docstring

    descriptions = []
    for func in functions:
        # Get the function signature
        sig = inspect.signature(func)

        # Get the function name
        func_name = func.__name__

        # Reconstruct the function definition
        func_def = f"def {func_name}{sig}:"

        # Get the docstring
        docstring = inspect.getdoc(func)
        if docstring:
            # Indent the docstring properly
            docstring_lines = docstring.split("\n")
            formatted_docstring = '    """' + docstring_lines[0]
            if len(docstring_lines) > 1:
                for line in docstring_lines[1:]:
                    formatted_docstring += "\n    " + line
            formatted_docstring += '"""'
        else:
            formatted_docstring = '    """No docstring available."""'

        # Combine function definition and docstring
        full_description = func_def + "\n" + formatted_docstring
        descriptions.append(full_description)

    return descriptions
