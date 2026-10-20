"""
FastAPI dependency injection providers for registry, storage, LLM, indexer, and RAG engine.
Supports dependency overrides in tests via FastAPI's app.dependency_overrides.
"""
from __future__ import annotations

from typing import Any
from fastapi import Depends

from src.storage.base import VectorStore
from src.storage.chroma_store import get_vector_store
from src.registry.registry import RepoRegistry, get_repo_registry
from src.registry.incremental import IncrementalIndexer
from src.llm.base import LLMProvider, get_llm_provider
from src.rag.engine import RAGEngine

# Module-level singletons (lazily initialized)
_GLOBAL_REGISTRY: RepoRegistry | None = None
_GLOBAL_STORAGE: VectorStore | None = None
_GLOBAL_LLM: LLMProvider | None = None


def get_registry() -> RepoRegistry:
    """Dependency provider for the repository registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = get_repo_registry()
    return _GLOBAL_REGISTRY


def get_storage() -> VectorStore:
    """Dependency provider for the vector database store."""
    global _GLOBAL_STORAGE
    if _GLOBAL_STORAGE is None:
        _GLOBAL_STORAGE = get_vector_store()
    return _GLOBAL_STORAGE


def get_llm() -> LLMProvider:
    """Dependency provider for the LLM provider."""
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        _GLOBAL_LLM = get_llm_provider()
    return _GLOBAL_LLM


def get_indexer(
    storage: VectorStore = Depends(get_storage),
    registry: RepoRegistry = Depends(get_registry),
) -> IncrementalIndexer:
    """Dependency provider for the IncrementalIndexer."""
    return IncrementalIndexer(vector_store=storage, registry=registry)


def get_rag_engine(
    storage: VectorStore = Depends(get_storage),
    llm: LLMProvider = Depends(get_llm),
    registry: RepoRegistry = Depends(get_registry),
) -> RAGEngine:
    """Dependency provider for the RAGEngine."""
    return RAGEngine(vector_store=storage, llm_provider=llm, registry=registry)


def set_global_dependencies(
    registry: RepoRegistry | None = None,
    storage: VectorStore | None = None,
    llm: LLMProvider | None = None,
) -> None:
    """Configure global dependencies (useful for initialization and scripts)."""
    global _GLOBAL_REGISTRY, _GLOBAL_STORAGE, _GLOBAL_LLM
    if registry is not None:
        _GLOBAL_REGISTRY = registry
    if storage is not None:
        _GLOBAL_STORAGE = storage
    if llm is not None:
        _GLOBAL_LLM = llm


def reset_global_dependencies() -> None:
    """Reset singletons so they reinitialize on next call."""
    global _GLOBAL_REGISTRY, _GLOBAL_STORAGE, _GLOBAL_LLM
    _GLOBAL_REGISTRY = None
    _GLOBAL_STORAGE = None
    _GLOBAL_LLM = None
