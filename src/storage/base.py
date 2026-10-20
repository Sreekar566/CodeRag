"""
Abstract base interface for vector storage and semantic retrieval.
Allows swapping vector database implementations (Chroma, in-memory, etc.)
without changing higher-level ingestion or RAG query logic.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize a repository or collection name to meet naming requirements:
    - 3-63 characters
    - Alphanumeric, underscores, hyphens
    - Starts and ends with alphanumeric
    """
    # Replace slashes and other non-alphanumeric/hyphen/underscore chars with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Strip leading/trailing non-alphanumeric chars
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)
    if len(sanitized) < 3:
        sanitized = f"col_{sanitized}".rstrip("_")
        if len(sanitized) < 3:
            sanitized = "codebase"
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip("_-")
    return sanitized


class VectorStore(ABC):
    """
    Abstract interface for vector database operations.
    """

    @abstractmethod
    def add_chunks(self, chunks: list[dict], collection_name: str | None = None) -> int:
        """
        Store or update code chunks with their vector embeddings and metadata.
        Returns the number of chunks stored.
        """
        pass

    @abstractmethod
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        collection_name: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Search for the top_k most semantically similar code chunks for a natural-language query.
        Returns a list of result dicts containing id, text, path, start_line, end_line,
        language, distance, similarity, metadata.
        """
        pass

    @abstractmethod
    def query_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Search for the top_k most similar code chunks using a raw embedding vector.
        """
        pass

    @abstractmethod
    def get(self, ids: list[str], collection_name: str | None = None) -> list[dict]:
        """
        Retrieve chunks by their IDs.
        """
        pass

    @abstractmethod
    def count(self, collection_name: str | None = None) -> int:
        """
        Return the total number of items stored in the collection.
        """
        pass

    @abstractmethod
    def clear(self, collection_name: str | None = None) -> None:
        """
        Remove all items from the collection.
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Delete items by ID or metadata filter.
        """
        pass

    @abstractmethod
    def list_collections(self) -> list[str]:
        """
        List all collection names in the store.
        """
        pass
