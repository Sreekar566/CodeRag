"""
Token counting, budget calculation, and text truncation utilities for RAG prompts.
Uses tiktoken (cl100k_base by default) with robust fallback heuristics.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = "cl100k_base"


@functools.lru_cache(maxsize=8)
def get_encoding(encoding_name: str = DEFAULT_ENCODING):
    """
    Get and cache a tiktoken encoding object. Returns None if tiktoken is unavailable.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        logger.warning("Failed to load tiktoken encoding '%s': %s. Falling back to heuristic counter.", encoding_name, exc)
        return None


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """
    Count the number of tokens in the given text.
    Uses tiktoken if available, otherwise falls back to a regex word/punctuation heuristic.
    """
    if not text:
        return 0

    enc = get_encoding(encoding_name)
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception as exc:
            logger.debug("tiktoken encoding failed: %s, falling back to heuristic", exc)

    # Fallback heuristic: words + punctuation / special characters
    # Code typically averages ~3.5 characters per token or ~1.3 tokens per whitespace-separated word
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(tokens))


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = DEFAULT_ENCODING,
    preserve_lines: bool = True,
) -> tuple[str, int]:
    """
    Truncate text so its token count does not exceed `max_tokens`.
    
    If `preserve_lines` is True, truncates at clean line boundaries when possible.
    Returns:
        tuple of (truncated_text, token_count)
    """
    if max_tokens <= 0:
        return "", 0

    current_tokens = count_tokens(text, encoding_name=encoding_name)
    if current_tokens <= max_tokens:
        return text, current_tokens

    enc = get_encoding(encoding_name)

    if preserve_lines and "\n" in text:
        lines = text.splitlines(keepends=True)
        kept_lines: list[str] = []
        accumulated_text = ""

        truncation_suffix = "\n... [truncated to fit token limit]"
        suffix_tokens = count_tokens(truncation_suffix, encoding_name=encoding_name)
        effective_limit = max(1, max_tokens - suffix_tokens)

        for line in lines:
            candidate = accumulated_text + line
            cand_tokens = count_tokens(candidate, encoding_name=encoding_name)
            if cand_tokens <= effective_limit:
                kept_lines.append(line)
                accumulated_text = candidate
            else:
                break

        if kept_lines:
            truncated = "".join(kept_lines).rstrip() + truncation_suffix
            actual_count = count_tokens(truncated, encoding_name=encoding_name)
            return truncated, actual_count

    # Token-level truncation if lines cannot be preserved or line-preserving produced nothing
    if enc is not None:
        try:
            token_ids = enc.encode(text, disallowed_special=())
            truncated_ids = token_ids[:max_tokens]
            decoded = enc.decode(truncated_ids)
            return decoded, len(truncated_ids)
        except Exception:
            pass

    # Heuristic fallback: approx 3.8 chars per token
    char_limit = int(max_tokens * 3.8)
    truncated = text[:char_limit]
    return truncated, count_tokens(truncated, encoding_name=encoding_name)


def estimate_chunk_tokens(chunk: dict[str, Any], encoding_name: str = DEFAULT_ENCODING) -> int:
    """
    Calculate the total token cost of a chunk as formatted for prompt context,
    including file header, line numbers, language tag, and markdown fences.
    """
    path = chunk.get("path") or chunk.get("metadata", {}).get("path", "unknown")
    start = chunk.get("start_line") or chunk.get("metadata", {}).get("start_line", 1)
    end = chunk.get("end_line") or chunk.get("metadata", {}).get("end_line", 1)
    lang = chunk.get("language") or chunk.get("metadata", {}).get("language", "")
    text = chunk.get("text", "").strip()

    header = f"[Source]\nFile: {path} (Lines {start}-{end})\nLanguage: {lang}\n```{lang}\n"
    footer = "\n```"
    return count_tokens(header + text + footer, encoding_name=encoding_name)
