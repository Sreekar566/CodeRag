"""
Endpoints for listing, inspecting, and deleting vector store collections.
"""
from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_storage
from src.api.models import (
    CollectionDetailResponse,
    CollectionListResponse,
    CollectionResponse,
    DeleteResponse,
)
from src.storage.base import VectorStore, sanitize_collection_name
from src.storage.chroma_store import ChromaVectorStore
from src.storage.memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])


def _get_sample_chunks(storage: VectorStore, collection_name: str, limit: int = 5) -> list[dict[str, Any]]:
    """Helper to safely retrieve a sample of indexed chunks from any vector store backend."""
    sanitized = sanitize_collection_name(collection_name)
    sample_items: list[dict[str, Any]] = []

    try:
        if isinstance(storage, InMemoryVectorStore):
            items = list(storage._get_collection(sanitized).values())[:limit]
            for it in items:
                meta = it.get("metadata", {})
                sample_items.append({
                    "id": it.get("id"),
                    "text": (it.get("text") or "")[:200],
                    "path": meta.get("path"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "language": meta.get("language"),
                })
        elif isinstance(storage, ChromaVectorStore):
            col = storage.get_or_create_collection(sanitized)
            raw = col.get(limit=limit)
            if raw and raw.get("ids"):
                ids = raw["ids"]
                docs = raw.get("documents") or [None] * len(ids)
                metas = raw.get("metadatas") or [{}] * len(ids)
                for chunk_id, doc, meta in zip(ids, docs, metas):
                    meta = meta or {}
                    sample_items.append({
                        "id": chunk_id,
                        "text": (doc or "")[:200],
                        "path": meta.get("path"),
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                        "language": meta.get("language"),
                    })
    except Exception as e:
        logger.warning(f"Could not retrieve sample chunks for {collection_name}: {e}")

    return sample_items


@router.get("", response_model=CollectionListResponse, summary="List all vector collections")
def list_collections(
    storage: VectorStore = Depends(get_storage),
) -> CollectionListResponse:
    """
    List all collections stored in the vector database and their total chunk count.
    """
    collection_names = storage.list_collections()
    results: list[CollectionResponse] = []
    for name in collection_names:
        try:
            count = storage.count(name)
        except Exception:
            count = 0
        results.append(CollectionResponse(name=name, chunk_count=count))

    return CollectionListResponse(collections=results, total=len(results))


@router.get("/{name}", response_model=CollectionDetailResponse, summary="Get collection details")
def get_collection(
    name: str,
    storage: VectorStore = Depends(get_storage),
) -> CollectionDetailResponse:
    """
    Get statistics and sample chunks for a specific vector collection.
    """
    sanitized = sanitize_collection_name(name)
    existing_collections = storage.list_collections()
    if sanitized not in existing_collections and name not in existing_collections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{name}' does not exist in vector store.",
        )

    matched_name = sanitized if sanitized in existing_collections else name
    chunk_count = storage.count(matched_name)
    sample_chunks = _get_sample_chunks(storage, matched_name, limit=5)

    return CollectionDetailResponse(
        name=matched_name,
        chunk_count=chunk_count,
        sample_chunks=sample_chunks,
    )


@router.delete("/{name}", response_model=DeleteResponse, summary="Delete a vector collection")
def delete_collection(
    name: str,
    storage: VectorStore = Depends(get_storage),
) -> DeleteResponse:
    """
    Delete a vector collection and all its embedded code chunks.
    """
    sanitized = sanitize_collection_name(name)
    existing_collections = storage.list_collections()
    if sanitized not in existing_collections and name not in existing_collections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{name}' not found.",
        )

    target_name = sanitized if sanitized in existing_collections else name
    try:
        storage.clear(target_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear collection '{target_name}': {e}",
        )

    return DeleteResponse(
        success=True,
        message=f"Collection '{target_name}' deleted.",
        target=target_name,
    )
