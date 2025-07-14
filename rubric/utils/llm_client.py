"""LLM client utility for making calls to LLMs compatible with the OpenAI's API."""

import base64
import os
from typing import Any, List, Union

from openai import OpenAI
from openai.types.chat import ChatCompletion

from rubric.constants import RUBRIC_DEFAULT_LLM


class LLMClient:
    """Client for making calls to LLM models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            api_key: LLM API key. If None, will try to get from OPENAI_API_KEY env var.
            model: The model to use for completions.
            base_url: Base URL for the API endpoint. If None, will try to get from
                OPENAI_BASE_URL env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LLM API key is required. Set OPENAI_API_KEY environment variable or pass "
                "api_key parameter."
            )

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.base_url:
            raise ValueError(
                "LLM base URL is required. Set OPENAI_BASE_URL environment variable or pass "
                "base_url parameter."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model or RUBRIC_DEFAULT_LLM

    def chat_completion(
        self,
        messages: Any,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Make a chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            temperature: Controls randomness in the response (0.0 to 2.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            The generated text response.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            if not response.choices:
                raise Exception("No response choices received from OpenAI API")

            return response.choices[0].message.content or ""

        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

    def simple_completion(
        self, prompt: str, temperature: float = 0.7, max_tokens: int | None = None, **kwargs: Any
    ) -> str:
        """Make a simple completion request with a single user message.

        Args:
            prompt: The prompt text to send to the model.
            temperature: Controls randomness in the response (0.0 to 2.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            The generated text response.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, temperature, max_tokens, **kwargs)

    def system_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Make a completion request with a system message and user message.

        Args:
            system_prompt: The system prompt to set the context.
            user_prompt: The user prompt to send to the model.
            temperature: Controls randomness in the response (0.0 to 2.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            The generated text response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat_completion(messages, temperature, max_tokens, **kwargs)

    def vision_completion(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Make a vision completion request with text prompt and multiple images.

        Args:
            prompt: The text prompt to send to the model.
            images: List of images. Each image can be:
                - File path (string) - will be read and base64 encoded
                - Base64 encoded string
                - Raw bytes - will be base64 encoded
            temperature: Controls randomness in the response (0.0 to 2.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            The generated text response.

        Raises:
            Exception: If the API call fails or image processing fails.
        """
        try:
            # Prepare the content list starting with the text prompt
            content: List[Any] = [{"type": "text", "text": prompt}]

            # Process each image
            for image in images:
                if isinstance(image, str):
                    # Check if it's a file path or already base64 encoded
                    if image.startswith("data:image/") or image.startswith("http"):
                        # It's already a data URL or HTTP URL
                        image_url = image
                    else:
                        # It's a file path, read and encode it
                        try:
                            with open(image, "rb") as image_file:
                                image_data = image_file.read()
                                image_base64 = base64.b64encode(image_data).decode("utf-8")
                                # Determine the image format from file extension
                                file_ext = image.lower().split(".")[-1]
                                if file_ext in ["jpg", "jpeg"]:
                                    mime_type = "image/jpeg"
                                elif file_ext == "png":
                                    mime_type = "image/png"
                                elif file_ext == "gif":
                                    mime_type = "image/gif"
                                elif file_ext == "webp":
                                    mime_type = "image/webp"
                                else:
                                    mime_type = "image/jpeg"  # Default fallback
                                image_url = f"data:{mime_type};base64,{image_base64}"
                        except FileNotFoundError:
                            raise Exception(f"Image file not found: {image}")
                        except Exception as e:
                            raise Exception(f"Error reading image file {image}: {str(e)}")

                elif isinstance(image, bytes):
                    # It's raw bytes, encode it as base64
                    image_base64 = base64.b64encode(image).decode("utf-8")
                    image_url = f"data:image/jpeg;base64,{image_base64}"  # Default to jpeg

                else:
                    raise Exception(f"Invalid image format: {type(image)}. Expected str or bytes.")

                # Add the image to content
                content.append({"type": "image_url", "image_url": {"url": image_url}})

            # Create the message with vision content
            messages = [{"role": "user", "content": content}]

            return self.chat_completion(messages, temperature, max_tokens, **kwargs)

        except Exception as e:
            raise Exception(f"Vision completion failed: {str(e)}")


def create_llm_client(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Create a new LLM client instance.

    Args:
        api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
        model: The model to use for completions.
        base_url: Base URL for the API endpoint. If None, will try to get from
            OPENAI_BASE_URL env var, then use default.

    Returns:
        A configured LLMClient instance.
    """
    return LLMClient(api_key=api_key, model=model, base_url=base_url)
