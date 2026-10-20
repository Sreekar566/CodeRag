"""
Prompt generation and context formatting for RAG code intelligence.
"""
from __future__ import annotations

from typing import Any

DEFAULT_SYSTEM_PROMPT = """You are an expert AI code intelligence assistant.
Your task is to answer questions about a codebase accurately and concisely, grounded strictly in the provided code context chunks.

Guidelines:
1. Ground every claim directly in the retrieved code. Do not hallucinate or assume files, functions, or configurations that are not shown in the context.
2. Always cite specific files and line numbers where the implementation lives using the format `path/to/file.ext:start-end` (for example: `app/auth/routes.py:12-35`).
3. If the provided code chunks do not contain enough information to fully answer the question, explicitly state what is known from the context and what information is missing.
"""


def format_code_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format a list of retrieved chunks into a clean, numbered context block for LLM prompts.
    """
    if not chunks:
        return "No relevant code context found."

    context_blocks: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        path = chunk.get("path") or chunk.get("metadata", {}).get("path", "unknown")
        start = chunk.get("start_line") or chunk.get("metadata", {}).get("start_line", 1)
        end = chunk.get("end_line") or chunk.get("metadata", {}).get("end_line", 1)
        lang = chunk.get("language") or chunk.get("metadata", {}).get("language", "")
        text = chunk.get("text", "").strip()

        block = (
            f"[Source {i}]\n"
            f"File: {path} (Lines {start}-{end})\n"
            f"Language: {lang}\n"
            f"```{lang}\n"
            f"{text}\n"
            f"```"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """
    Construct the complete user prompt containing the question and formatted code context.
    """
    context_str = format_code_context(chunks)
    return (
        f"Codebase Context:\n"
        f"-----------------\n"
        f"{context_str}\n"
        f"-----------------\n\n"
        f"User Question: {query}\n\n"
        f"Please provide an accurate, grounded answer with citations to specific file paths and line ranges."
    )
