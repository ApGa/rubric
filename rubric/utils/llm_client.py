"""LLM client utility for making calls to LLMs using LiteLLM."""

import asyncio
import base64
import json
import os
from typing import Any, List, Union

from litellm import acompletion, completion

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
        # Allow lazy configuration; only enforce presence at call time
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        self.model = model or RUBRIC_DEFAULT_LLM

    def chat_completion(
        self,
        messages: Any,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
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
            response = completion(
                **self._build_call_kwargs(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    **kwargs,
                )
            )
            return self._extract_response_text(response, **kwargs)
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")

    async def achat_completion(
        self,
        messages: Any,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Make an async chat completion request."""
        try:
            response = await acompletion(
                **self._build_call_kwargs(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    **kwargs,
                )
            )
            return self._extract_response_text(response, **kwargs)
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")

    def simple_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Make a simple completion request with a single user message.

        Args:
            prompt: The prompt text to send to the model.
            temperature: Controls randomness in the response (0.0 to 2.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments to pass to the LLM API.

        Returns:
            The generated text response.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(
            messages, temperature, max_tokens, reasoning_effort=reasoning_effort, **kwargs
        )

    async def asimple_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Make an async completion request with a single user message."""
        messages = [{"role": "user", "content": prompt}]
        return await self.achat_completion(
            messages, temperature, max_tokens, reasoning_effort=reasoning_effort, **kwargs
        )

    def system_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Make a completion request with a system message and user message.

        Args:
            system_prompt: The system prompt to set the context.
            user_prompt: The user prompt to send to the model.
            temperature: Controls randomness in the response (0.0 to 2.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments to pass to the LLM API.

        Returns:
            The generated text response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat_completion(
            messages, temperature, max_tokens, reasoning_effort=reasoning_effort, **kwargs
        )

    async def asystem_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Make an async completion request with system and user messages."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.achat_completion(
            messages, temperature, max_tokens, reasoning_effort=reasoning_effort, **kwargs
        )

    def vision_completion(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
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
            **kwargs: Additional arguments to pass to the LLM API.

        Returns:
            The generated text response.

        Raises:
            Exception: If the API call fails or image processing fails.
        """
        try:
            messages = [{"role": "user", "content": self._build_vision_content(prompt, images)}]
            return self.chat_completion(
                messages, temperature, max_tokens, reasoning_effort=reasoning_effort, **kwargs
            )
        except Exception as e:
            raise Exception(f"Vision completion failed: {str(e)}")

    async def avision_completion(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Make an async vision completion request."""
        try:
            content = await asyncio.to_thread(self._build_vision_content, prompt, images)
            messages = [{"role": "user", "content": content}]
            return await self.achat_completion(
                messages, temperature, max_tokens, reasoning_effort=reasoning_effort, **kwargs
            )
        except Exception as e:
            raise Exception(f"Vision completion failed: {str(e)}")

    def _build_call_kwargs(
        self,
        messages: Any,
        temperature: float,
        max_tokens: int | None,
        reasoning_effort: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build LiteLLM completion kwargs shared by sync and async calls."""
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if self.base_url:
            call_kwargs["api_base"] = self.base_url
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if reasoning_effort:
            call_kwargs["reasoning_effort"] = reasoning_effort

        call_kwargs.update(kwargs)
        return call_kwargs

    def _extract_response_text(self, response: Any, **kwargs: Any) -> str:
        """Extract plain text from OpenAI-compatible sync or async responses."""
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")

        if not choices:
            raise Exception("No response choices received from LLM API")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None) or (
            first_choice.get("message") if isinstance(first_choice, dict) else None
        )
        if message is not None:
            response_format = kwargs.get("response_format") if isinstance(kwargs, dict) else None
            if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
                parsed = getattr(message, "parsed", None) or (
                    message.get("parsed") if isinstance(message, dict) else None
                )
                if parsed is not None:
                    return json.dumps(parsed)

            content = getattr(message, "content", None) or (
                message.get("content") if isinstance(message, dict) else None
            )
            if content is not None:
                if not isinstance(content, str):
                    try:
                        return json.dumps(content)
                    except Exception:
                        pass
                return str(content)

        text = getattr(first_choice, "text", None) or (
            first_choice.get("text") if isinstance(first_choice, dict) else None
        )
        if text is not None:
            return str(text)

        raise Exception("Unsupported response format from LLM API")

    def _build_vision_content(self, prompt: str, images: List[Union[str, bytes]]) -> List[Any]:
        """Prepare mixed text/image message content for vision requests."""
        content: List[Any] = [{"type": "text", "text": prompt}]

        for image in images:
            if isinstance(image, str):
                if image.startswith("data:image/") or image.startswith("http"):
                    image_url = image
                else:
                    try:
                        with open(image, "rb") as image_file:
                            image_data = image_file.read()
                            image_base64 = base64.b64encode(image_data).decode("utf-8")
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
                                mime_type = "image/jpeg"
                            image_url = f"data:{mime_type};base64,{image_base64}"
                    except FileNotFoundError:
                        raise Exception(f"Image file not found: {image}")
                    except Exception as e:
                        raise Exception(f"Error reading image file {image}: {str(e)}")
            elif isinstance(image, bytes):
                image_base64 = base64.b64encode(image).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{image_base64}"
            else:
                raise Exception(f"Invalid image format: {type(image)}. Expected str or bytes.")

            content.append({"type": "image_url", "image_url": {"url": image_url}})

        return content


def create_llm_client(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Create a new LLM client instance.

    Args:
        api_key: LLM API key. If None, will try to get from OPENAI_API_KEY env var.
        model: The model to use for completions.
        base_url: Base URL for the API endpoint. If None, will try to get from
            OPENAI_BASE_URL env var, then use default.

    Returns:
        A configured LLMClient instance.
    """
    return LLMClient(api_key=api_key, model=model, base_url=base_url)
