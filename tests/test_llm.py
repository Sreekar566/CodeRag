"""
Unit tests for LLM provider abstraction, FakeLLMProvider, and GeminiLLMProvider.
All unit tests run offline without network access or real API keys.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from src.llm.base import LLMResponse, get_llm_provider
from src.llm.fake import FakeLLMProvider
from src.llm.gemini import GeminiLLMProvider


class TestFakeLLMProvider:
    def test_default_deterministic_response(self):
        provider = FakeLLMProvider()
        resp = provider.generate("How do I configure database?")
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Deterministic answer from FakeLLMProvider."
        assert resp.model == "fake-model"
        assert provider.call_count == 1
        assert len(provider.history) == 1

    def test_canned_response_string(self):
        canned = "User login is implemented in app/auth/routes.py:15-30."
        provider = FakeLLMProvider(canned_response=canned)
        resp = provider.generate("Where is login?")
        assert resp.content == canned
        assert provider.complete("Where is login?") == canned
        assert provider.call_count == 2

    def test_canned_response_list_cycle(self):
        provider = FakeLLMProvider(responses=["Answer 1", "Answer 2"])
        assert provider.generate("q1").content == "Answer 1"
        assert provider.generate("q2").content == "Answer 2"
        assert provider.generate("q3").content == "Answer 1"
        assert provider.call_count == 3

    def test_canned_response_dict_lookup(self):
        provider = FakeLLMProvider(
            responses={
                "auth": "Auth is in app/auth/routes.py:1-50",
                "email": "Email helper is in app/email.py:10-25",
            }
        )
        assert "app/auth/routes.py" in provider.generate("How does auth work?").content
        assert "app/email.py" in provider.generate("Where is email sent?").content
        assert provider.generate("unknown question").content == "Default canned answer."

    def test_response_fn_callback(self):
        provider = FakeLLMProvider(response_fn=lambda p, s: f"Custom answer for: {p.strip()}")
        resp = provider.generate("What is this repo?")
        assert resp.content == "Custom answer for: What is this repo?"

    def test_context_extraction_from_prompt(self):
        provider = FakeLLMProvider()
        prompt = (
            "Context:\n"
            "File: app/auth/routes.py (Lines 10-35)\n"
            "```python\ndef login(): pass\n```\n"
            "Question: Where is login?"
        )
        resp = provider.generate(prompt)
        assert "app/auth/routes.py:10-35" in resp.content

    def test_reset_history(self):
        provider = FakeLLMProvider()
        provider.generate("test 1")
        provider.generate("test 2")
        assert provider.call_count == 2
        assert len(provider.history) == 2

        provider.reset()
        assert provider.call_count == 0
        assert len(provider.history) == 0


class TestGetLLMProviderFactory:
    def test_explicit_fake_provider(self):
        provider = get_llm_provider("fake")
        assert isinstance(provider, FakeLLMProvider)

    def test_env_var_fake_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "fake")
        provider = get_llm_provider()
        assert isinstance(provider, FakeLLMProvider)

    def test_env_var_gemini_without_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            get_llm_provider()

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider: 'unknown'"):
            get_llm_provider("unknown")

    def test_default_to_fake_when_no_env_or_key(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = get_llm_provider()
        assert isinstance(provider, FakeLLMProvider)


class TestGeminiLLMProvider:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiLLMProvider()

    def test_mock_client_generate(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded answer from Gemini."
        mock_response.finish_reason = "STOP"
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 70

        mock_client.models.generate_content.return_value = mock_response

        provider = GeminiLLMProvider(api_key="mock_test_key", client=mock_client)
        resp = provider.generate("Explain authentication", system_instruction="Be concise")

        assert resp.content == "Grounded answer from Gemini."
        assert resp.prompt_tokens == 50
        assert resp.completion_tokens == 20
        assert resp.total_tokens == 70

        mock_client.models.generate_content.assert_called_once()
        args, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["model"] == "gemini-3.8-flash"
        assert kwargs["contents"] == "Explain authentication"
        assert kwargs["config"].system_instruction == "Be concise"
