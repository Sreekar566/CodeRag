"""
Lightweight in-memory vector store implementing the VectorStore interface.
Useful for fast testing, ephemeral runs, or swappable backend validation.
"""
from __future__ import annotations

from typing import Any, Callable
import numpy as np

from src.storage.base import VectorStore, sanitize_collection_name


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store using numpy for cosine similarity calculation.
    """

    def __init__(
        self,
        default_collection: str = "codebase",
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self.default_collection = default_collection
        self._embed_fn = embed_fn
        # Structure: {col_name: {chunk_id: {"text": str, "embedding": list[float], "metadata": dict}}}
        self._collections: dict[str, dict[str, dict[str, Any]]] = {}

    def _get_collection(self, collection_name: str | None) -> dict[str, dict[str, Any]]:
        name = sanitize_collection_name(collection_name or self.default_collection)
        if name not in self._collections:
            self._collections[name] = {}
        return self._collections[name]

    def add_chunks(self, chunks: list[dict], collection_name: str | None = None) -> int:
        if not chunks:
            return 0

        col = self._get_collection(collection_name)

        # Check for missing embeddings
        missing = [c for c in chunks if "embedding" not in c or not c["embedding"]]
        if missing:
            if self._embed_fn is not None:
                for c in missing:
                    c["embedding"] = self._embed_fn(c["text"])
            else:
                from src.embedding.embedder import embed_chunks
                embed_chunks(missing)

        added = 0
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id")
            if not chunk_id:
                path = chunk.get("path", "unknown")
                start = chunk.get("start_line", 0)
                end = chunk.get("end_line", 0)
                chunk_id = f"{path}:{start}-{end}"
            chunk_id = str(chunk_id)

            meta = {}
            for k in ("path", "start_line", "end_line", "language"):
                if k in chunk and chunk[k] is not None:
                    meta[k] = chunk[k]
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                for k, v in chunk["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v

            col[chunk_id] = {
                "id": chunk_id,
                "text": chunk.get("text", ""),
                "embedding": chunk["embedding"],
                "metadata": meta,
            }
            added += 1

        return added

    def query_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        col = self._get_collection(collection_name)
        if not col:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        candidates = []
        for item in col.values():
            meta = item["metadata"]
            if where:
                matches = all(meta.get(k) == v for k, v in where.items())
                if not matches:
                    continue

            d_vec = np.array(item["embedding"], dtype=np.float32)
            d_norm = np.linalg.norm(d_vec)
            if d_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
            dist = max(0.0, 1.0 - sim)
            candidates.append((sim, dist, item))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_results = candidates[:top_k]

        results = []
        for sim, dist, item in top_results:
            meta = item["metadata"]
            results.append({
                "id": item["id"],
                "text": item["text"],
                "path": meta.get("path"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "language": meta.get("language"),
                "distance": round(dist, 4),
                "similarity": round(sim, 4),
                "metadata": meta,
            })
        return results

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        collection_name: str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
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

    def get(self, ids: list[str], collection_name: str | None = None) -> list[dict]:
        col = self._get_collection(collection_name)
        results = []
        for item_id in ids:
            if item_id in col:
                item = col[item_id]
                meta = item["metadata"]
                results.append({
                    "id": item["id"],
                    "text": item["text"],
                    "path": meta.get("path"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "language": meta.get("language"),
                    "metadata": meta,
                })
        return results

    def count(self, collection_name: str | None = None) -> int:
        return len(self._get_collection(collection_name))

    def clear(self, collection_name: str | None = None) -> None:
        name = sanitize_collection_name(collection_name or self.default_collection)
        self._collections.pop(name, None)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        col = self._get_collection(collection_name)
        if ids:
            for item_id in ids:
                col.pop(item_id, None)
        if where:
            to_remove = [
                k for k, v in col.items()
                if all(v["metadata"].get(mk) == mv for mk, mv in where.items())
            ]
            for item_id in to_remove:
                col.pop(item_id, None)

    def list_collections(self) -> list[str]:
        return list(self._collections.keys())
