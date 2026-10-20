"""
RAG (Retrieval-Augmented Generation) package for codebase intelligence.
"""
from src.rag.citations import Citation, extract_citations_from_text, citations_from_chunks
from src.rag.prompt import DEFAULT_SYSTEM_PROMPT, build_rag_prompt, format_code_context
from src.rag.engine import RAGEngine, RAGResponse, ask_codebase

__all__ = [
    "Citation",
    "extract_citations_from_text",
    "citations_from_chunks",
    "DEFAULT_SYSTEM_PROMPT",
    "build_rag_prompt",
    "format_code_context",
    "RAGEngine",
    "RAGResponse",
    "ask_codebase",
]
