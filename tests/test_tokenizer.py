"""
Unit tests for token counting, estimation, and truncation utilities.
"""
from unittest.mock import patch
import pytest

from src.rag.tokenizer import (
    count_tokens,
    truncate_text_to_tokens,
    estimate_chunk_tokens,
    get_encoding,
)


class TestTokenizer:
    def test_count_tokens_empty(self):
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    def test_count_tokens_basic(self):
        text = "def calculate_total(items):\n    return sum(item.price for item in items)"
        tokens = count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_fallback(self):
        with patch("src.rag.tokenizer.get_encoding", return_value=None):
            text = "def hello(): print('world')"
            fallback_count = count_tokens(text)
            assert fallback_count > 0
            assert isinstance(fallback_count, int)

    def test_truncate_text_within_budget(self):
        text = "short text"
        truncated, count = truncate_text_to_tokens(text, max_tokens=100)
        assert truncated == text
        assert count <= 100

    def test_truncate_text_exceeds_budget_preserve_lines(self):
        lines = [f"line {i} = {i * 10}" for i in range(50)]
        text = "\n".join(lines)
        initial_tokens = count_tokens(text)
        assert initial_tokens > 50

        truncated, count = truncate_text_to_tokens(text, max_tokens=30, preserve_lines=True)
        assert count <= 35  # Allows leeway for truncation suffix
        assert "[truncated to fit token limit]" in truncated
        assert "line 0 =" in truncated
        assert len(truncated) < len(text)

    def test_truncate_text_token_level_when_no_lines(self):
        text = "word " * 100
        truncated, count = truncate_text_to_tokens(text, max_tokens=20, preserve_lines=False)
        assert count <= 20
        assert len(truncated) < len(text)

    def test_truncate_text_zero_or_negative(self):
        text = "some text"
        truncated, count = truncate_text_to_tokens(text, max_tokens=0)
        assert truncated == ""
        assert count == 0

    def test_estimate_chunk_tokens(self):
        chunk = {
            "path": "app/auth/routes.py",
            "start_line": 10,
            "end_line": 20,
            "language": "python",
            "text": "def login():\n    return 'login'",
        }
        tokens = estimate_chunk_tokens(chunk)
        # Should be greater than the raw text tokens due to formatting overhead
        raw_text_tokens = count_tokens(chunk["text"])
        assert tokens > raw_text_tokens
        assert tokens >= 20

    def test_estimate_chunk_tokens_from_nested_metadata(self):
        chunk = {
            "metadata": {
                "path": "config.py",
                "start_line": 1,
                "end_line": 5,
                "language": "python",
            },
            "text": "SECRET_KEY = 'secret'",
        }
        tokens = estimate_chunk_tokens(chunk)
        assert tokens > count_tokens(chunk["text"])
