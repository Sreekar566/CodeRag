"""
Coordinates repository ingestion, file filtering, chunking, embedding, and vector storage.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from src.ingestion.github_client import get_repo_tree, get_file_content
from src.ingestion.filters import filter_files
from src.ingestion.chunker import chunk_text
from src.embedding.embedder import embed_chunks
from src.storage.base import VectorStore, sanitize_collection_name
from src.storage.chroma_store import get_vector_store

logger = logging.getLogger(__name__)


def index_files(
    files: list[dict[str, str]],
    vector_store: VectorStore | None = None,
    collection_name: str | None = None,
    embedder_model: Any = None,
    chunk_fn: Callable[[str, str], list[dict]] = chunk_text,
) -> dict[str, Any]:
    """
    Ingest, chunk, embed, and store a list of file contents.
    Each file dict should have 'path' and 'content'.
    """
    if vector_store is None:
        vector_store = get_vector_store()

    all_chunks: list[dict] = []
    for f in files:
        path = f.get("path", "")
        content = f.get("content", "")
        if not path or not content:
            continue
        file_chunks = chunk_fn(content, path)
        all_chunks.extend(file_chunks)

    if not all_chunks:
        return {
            "collection": sanitize_collection_name(collection_name or "codebase"),
            "files_processed": len(files),
            "chunks_stored": 0,
        }

    # Generate embeddings
    embedded_chunks = embed_chunks(all_chunks, model=embedder_model)

    # Store in vector database
    stored_count = vector_store.add_chunks(embedded_chunks, collection_name=collection_name)

    return {
        "collection": sanitize_collection_name(collection_name or "codebase"),
        "files_processed": len(files),
        "chunks_stored": stored_count,
    }


def index_repository(
    owner: str,
    repo: str,
    token: str | None = None,
    vector_store: VectorStore | None = None,
    collection_name: str | None = None,
    branch: str = "main",
    embedder_model: Any = None,
) -> dict[str, Any]:
    """
    Full pipeline to index a GitHub repository into the vector database.
    1. Fetch repo file tree via GitHub REST API
    2. Filter out non-code/noise files
    3. Fetch content for each filtered file
    4. Chunk text into overlapping code chunks with line numbers & language metadata
    5. Compute vector embeddings
    6. Store chunks in vector DB
    """
    if collection_name is None:
        collection_name = f"{owner}_{repo}"
    collection_name = sanitize_collection_name(collection_name)

    if vector_store is None:
        vector_store = get_vector_store()

    # Step 1: List all files
    tree = get_repo_tree(owner, repo, token, branch=branch)

    # Step 2: Apply filter blocklist
    filtered_items = filter_files(tree)

    # Step 3: Fetch contents and chunk
    files: list[dict[str, str]] = []
    skipped_files: list[str] = []

    for item in filtered_items:
        path = item.get("path", "")
        if not path:
            continue
        try:
            content = get_file_content(owner, repo, path, token)
            files.append({"path": path, "content": content})
        except Exception as e:
            logger.warning(f"Failed to fetch content for {path}: {e}")
            skipped_files.append(path)

    # Step 4-6: Chunk, embed, store
    result = index_files(
        files=files,
        vector_store=vector_store,
        collection_name=collection_name,
        embedder_model=embedder_model,
    )

    result.update({
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "total_tree_files": len(tree),
        "filtered_files": len(filtered_items),
        "skipped_files": skipped_files,
    })

    return result
