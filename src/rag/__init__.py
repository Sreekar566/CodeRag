"""
RAG (Retrieval-Augmented Generation) package for codebase intelligence.
Includes token budgeting, re-ranking, context assembly, prompt construction, LLM generation, and citations.
"""
from src.rag.citations import Citation, extract_citations_from_text, citations_from_chunks
from src.rag.prompt import DEFAULT_SYSTEM_PROMPT, build_rag_prompt, format_code_context
from src.rag.engine import RAGEngine, RAGResponse, ask_codebase
from src.rag.tokenizer import (
    count_tokens,
    truncate_text_to_tokens,
    estimate_chunk_tokens,
    get_encoding,
)
from src.rag.reranker import (
    ReRanker,
    rerank_chunks,
    merge_overlapping_chunks,
    extract_query_keywords,
    compute_lexical_score,
    compute_path_score,
    compute_definition_score,
)
from src.rag.context_assembler import (
    ContextAssembler,
    ContextAssemblyResult,
    assemble_context,
    reorder_for_attention,
    DEFAULT_MAX_CONTEXT_TOKENS,
)

__all__ = [
    # Citations
    "Citation",
    "extract_citations_from_text",
    "citations_from_chunks",
    # Prompt
    "DEFAULT_SYSTEM_PROMPT",
    "build_rag_prompt",
    "format_code_context",
    # Engine
    "RAGEngine",
    "RAGResponse",
    "ask_codebase",
    # Tokenizer
    "count_tokens",
    "truncate_text_to_tokens",
    "estimate_chunk_tokens",
    "get_encoding",
    # Re-ranking
    "ReRanker",
    "rerank_chunks",
    "merge_overlapping_chunks",
    "extract_query_keywords",
    "compute_lexical_score",
    "compute_path_score",
    "compute_definition_score",
    # Context Assembly
    "ContextAssembler",
    "ContextAssemblyResult",
    "assemble_context",
    "reorder_for_attention",
    "DEFAULT_MAX_CONTEXT_TOKENS",
]
