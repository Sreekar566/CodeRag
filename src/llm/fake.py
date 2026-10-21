"""
Deterministic fake LLM provider for unit tests and offline evaluation.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from src.llm.base import LLMProvider, LLMResponse


class FakeLLMProvider(LLMProvider):
    """
    Deterministic canned LLM provider for unit tests.
    Does not require network access, API keys, or GPU/CPU model inference.
    """

    def __init__(
        self,
        canned_response: str | None = None,
        responses: list[str] | dict[str, str] | None = None,
        response_fn: Callable[[str, str | None], str] | None = None,
        model_name: str = "fake-model",
    ):
        self.canned_response = canned_response
        self.responses = responses
        self.response_fn = response_fn
        self.model_name = model_name
        self.call_count = 0
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset call count and invocation history."""
        self.call_count = 0
        self.history.clear()

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_count += 1
        record = {
            "prompt": prompt,
            "system_instruction": system_instruction,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "kwargs": kwargs,
        }
        self.history.append(record)

        content: str
        if self.response_fn is not None:
            content = self.response_fn(prompt, system_instruction)
        elif self.canned_response is not None:
            content = self.canned_response
        elif isinstance(self.responses, list) and self.responses:
            idx = (self.call_count - 1) % len(self.responses)
            content = self.responses[idx]
        elif isinstance(self.responses, dict):
            # Check prompt against keys
            matched_key = next((k for k in self.responses if k.lower() in prompt.lower()), None)
            content = self.responses[matched_key] if matched_key else "Default canned answer."
        else:
            prompt_lower = prompt.lower()
            if "user question:" in prompt_lower:
                question_part = prompt_lower.split("user question:")[1].split("please provide")[0]
            elif "question:" in prompt_lower:
                question_part = prompt_lower.split("question:")[1]
            else:
                question_part = prompt_lower

            if "codebase context:" in prompt_lower:
                # Extract between context boundaries
                ctx_split = prompt_lower.split("codebase context:")[1]
                context_part = ctx_split.split("user question:")[0]
            else:
                context_part = prompt_lower

            # Check if out-of-scope probe keywords exist in question but are absent in retrieved context
            is_unanswerable = False
            for probe in ["stripe", "payment", "kubernetes", "helm", "graphql", "celery", "kafka"]:
                if probe in question_part and probe not in context_part:
                    is_unanswerable = True
                    break

            if is_unanswerable:
                content = "Based on the provided codebase context, this feature is not found and does not appear to be implemented in the repository."
            else:
                # Deterministic extraction of context file/lines from prompt if present
                # Look for lines like "File: {path} (Lines {start}-{end})" or "File: {path}"
                match = re.search(r"File:\s*([^\s\n()]+)(?:\s*\(Lines\s*(\d+)-(\d+)\))?", prompt)
                if match:
                    path = match.group(1)
                    start = match.group(2)
                    end = match.group(3)
                    if start and end:
                        content = f"Based on the provided codebase context, this is implemented in {path}:{start}-{end}."
                    else:
                        content = f"Based on the provided codebase context, this is implemented in {path}."
                else:
                    content = "Deterministic answer from FakeLLMProvider."

        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split())
        return LLMResponse(
            content=content,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            metadata={"call_index": self.call_count},
        )
