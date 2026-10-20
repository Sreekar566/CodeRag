"""
Gemini LLM provider using the modern google-genai SDK.
"""
from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

from src.llm.base import LLMProvider, LLMResponse

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class GeminiLLMProvider(LLMProvider):
    """
    LLM provider backed by Google's Gemini models via google-genai SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key and client is None:
            raise ValueError(
                "GEMINI_API_KEY environment variable or api_key parameter is required "
                "for GeminiLLMProvider."
            )
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._client = client

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if max_output_tokens:
            config_kwargs["max_output_tokens"] = max_output_tokens

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        content = response.text or ""

        # Extract token usage metadata if available
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
            total_tokens = getattr(response.usage_metadata, "total_token_count", None)

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata={"raw_response": response},
        )
