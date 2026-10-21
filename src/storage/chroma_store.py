"""
Chroma vector database implementation of the VectorStore interface.
Provides local persistent storage in ./data/chroma/ or ephemeral in-memory storage.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import chromadb

from src.storage.base import VectorStore, sanitize_collection_name

DEFAULT_CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
DEFAULT_COLLECTION_NAME = "codebase"
BATCH_SIZE = 500


class ChromaVectorStore(VectorStore):
    """
    Chroma-backed vector store for code chunks and embeddings.
    """

    def __init__(
        self,
        persist_directory: str | None = DEFAULT_CHROMA_PATH,
        default_collection: str = DEFAULT_COLLECTION_NAME,
        embed_fn: Callable[[str], list[float]] | None = None,
        client: Any = None,
    ):
        self.persist_directory = persist_directory
        self.default_collection = default_collection
        self._embed_fn = embed_fn

        if client is not None:
            self._client = client
        elif persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

    def get_or_create_collection(self, collection_name: str | None = None) -> Any:
        col_name = sanitize_collection_name(collection_name or self.default_collection)
        return self._client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add_chunks(self, chunks: list[dict], collection_name: str | None = None) -> int:
        """
        Store or update code chunks with their vector embeddings and metadata.
        Chunks lacking embeddings will be embedded automatically.
        """
        if not chunks:
            return 0

        col = self.get_or_create_collection(collection_name)

        # Ensure all chunks have embeddings
        missing_embeddings = [c for c in chunks if "embedding" not in c or not c["embedding"]]
        if missing_embeddings:
            if self._embed_fn is not None:
                for c in missing_embeddings:
                    c["embedding"] = self._embed_fn(c["text"])
            else:
                from src.embedding.embedder import embed_chunks
                embed_chunks(missing_embeddings)

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        seen_ids: set[str] = set()
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id")
            if not chunk_id:
                path = chunk.get("path", "unknown")
                start = chunk.get("start_line", 0)
                end = chunk.get("end_line", 0)
                chunk_id = f"{path}:{start}-{end}"

            # Disambiguate if duplicate in same batch
            if chunk_id in seen_ids:
                chunk_id = f"{chunk_id}#{i}"
            seen_ids.add(chunk_id)

            ids.append(str(chunk_id))
            documents.append(chunk.get("text", ""))
            embeddings.append(chunk["embedding"])

            # Build metadata dict with supported types only
            meta: dict[str, str | int | float | bool] = {}
            for k in ("path", "start_line", "end_line", "language"):
                val = chunk.get(k)
                if val is not None and isinstance(val, (str, int, float, bool)):
                    meta[k] = val

            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                for k, v in chunk["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v

            # Chroma requires every metadata dict to be non-empty
            if not meta:
                meta["source"] = "codebase"

            metadatas.append(meta)

        # Batch upsert to stay safely within limits
        total_added = 0
        for idx in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[idx : idx + BATCH_SIZE]
            batch_docs = documents[idx : idx + BATCH_SIZE]
            batch_embs = embeddings[idx : idx + BATCH_SIZE]
            batch_metas = metadatas[idx : idx + BATCH_SIZE]

            col.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            total_added += len(batch_ids)

        return total_added

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        collection_name: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Search for chunks semantically matching the query text.
        """
        if not query_text:
            return []

        if self._embed_fn is not None:
            query_embedding = self._embed_fn(query_text)
        else:
            from src.embedding.embedder import embed_query
            query_embedding = embed_query(query_text)

        return self.query_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=collection_name,
            where=where,
        )

    def query_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Search for chunks closest to the given embedding vector.
        """
        col = self.get_or_create_collection(collection_name)
        total_items = col.count()
        if total_items == 0:
            return []

        n_results = min(top_k, total_items)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        raw = col.query(**kwargs)
        return self._format_results(raw)

    def get(self, ids: list[str], collection_name: str | None = None) -> list[dict]:
        """
        Retrieve stored chunks by ID.
        """
        if not ids:
            return []

        col = self.get_or_create_collection(collection_name)
        raw = col.get(ids=ids)
        results: list[dict] = []
        if not raw or not raw.get("ids"):
            return results

        r_ids = raw["ids"]
        documents = raw.get("documents") or [None] * len(r_ids)
        metadatas = raw.get("metadatas") or [{}] * len(r_ids)

        for doc_id, doc, meta in zip(r_ids, documents, metadatas):
            meta = meta or {}
            results.append({
                "id": doc_id,
                "text": doc,
                "path": meta.get("path"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "language": meta.get("language"),
                "metadata": meta,
            })
        return results

    def count(self, collection_name: str | None = None) -> int:
        """
        Return the total count of stored chunks.
        """
        return self.get_or_create_collection(collection_name).count()

    def clear(self, collection_name: str | None = None) -> None:
        """
        Clear all stored chunks in the collection.
        """
        col_name = sanitize_collection_name(collection_name or self.default_collection)
        try:
            self._client.delete_collection(col_name)
        except Exception:
            pass

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Delete items by ID or metadata filter.
        """
        col = self.get_or_create_collection(collection_name)
        kwargs: dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        if kwargs:
            col.delete(**kwargs)

    def list_collections(self) -> list[str]:
        """
        List all collection names in the store.
        """
        cols = self._client.list_collections()
        return [c.name if hasattr(c, "name") else str(c) for c in cols]

    def _format_results(self, raw: dict) -> list[dict]:
        results: list[dict] = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        documents = raw.get("documents", [[]])[0] if raw.get("documents") else [None] * len(ids)
        metadatas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else [{}] * len(ids)
        distances = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(ids)

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            meta = meta or {}
            # Cosine distance in Chroma: dist in [0, 2]; similarity = 1 - dist
            dist_val = float(dist) if dist is not None else 0.0
            similarity = round(1.0 - dist_val, 4)
            results.append({
                "id": doc_id,
                "text": doc,
                "path": meta.get("path"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "language": meta.get("language"),
                "distance": round(dist_val, 4),
                "similarity": similarity,
                "metadata": meta,
            })
        return results


def get_vector_store(
    persist_directory: str | None = DEFAULT_CHROMA_PATH,
    default_collection: str = DEFAULT_COLLECTION_NAME,
    **kwargs: Any,
) -> VectorStore:
    """
    Factory helper to instantiate the default vector store.
    """
    return ChromaVectorStore(
        persist_directory=persist_directory,
        default_collection=default_collection,
        **kwargs,
    )
