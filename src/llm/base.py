"""
Abstract base interface for LLM providers and factory loader.
Allows swapping between Gemini, a fake provider for testing, and future providers.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Represents a generated completion from an LLM provider."""
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


class LLMProvider(ABC):
    """Abstract interface for LLM operations."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a text response for the given prompt.
        """
        pass

    def complete(
        self,
        prompt: str,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience helper returning just the text completion."""
        response = self.generate(prompt, system_instruction=system_instruction, **kwargs)
        return response.content


def get_llm_provider(
    provider: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory helper to instantiate an LLM provider.
    Selects provider based on:
    1. Explicit provider parameter
    2. LLM_PROVIDER environment variable
    3. Defaults to 'gemini' if GEMINI_API_KEY is present, else 'fake'
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER")

    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "fake"

    provider_normalized = provider.strip().lower()

    if provider_normalized == "fake":
        from src.llm.fake import FakeLLMProvider
        return FakeLLMProvider(**kwargs)
    elif provider_normalized == "gemini":
        from src.llm.gemini import GeminiLLMProvider
        return GeminiLLMProvider(**kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported providers are 'fake' and 'gemini'."
        )
