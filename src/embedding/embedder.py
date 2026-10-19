# Code-specialized embedding model (not a general-purpose text model) —
# see project_log.md for the reasoning: retrieval quality for code questions
# should benefit from a model actually trained on code.
# Originally jinaai/jina-embeddings-v2-base-code, but its trust_remote_code
# custom loading path deadlocked on this machine with no diagnosable error
# (see project_log.md) — switched to a standard-architecture code model
# that sentence-transformers loads natively, no custom remote code needed.
from __future__ import annotations
from typing import Any
from sentence_transformers import SentenceTransformer
from src.ingestion.chunker import detect_language

MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazily load the SentenceTransformer model to avoid expensive downloads/loads
    at module import time. Caches the loaded model instance.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list[dict], model: Any = None) -> list[dict]:
    """
    Takes the chunk dicts produced by chunk_text() and adds an "embedding"
    vector to each one, in place. Also ensures each chunk has "language" metadata.

    This is the swappable interface: any other embedding backend (a hosted
    API, a different local model) should implement the same signature —
    take chunks in, return the same chunks with "embedding" added.
    """
    if not chunks:
        return chunks

    # Ensure language metadata is attached to every chunk
    for c in chunks:
        if "language" not in c and "path" in c:
            c["language"] = detect_language(c["path"])

    # step 1: pull out just the text from each chunk into a single list —
    # encoding all chunks in one batched call is much faster than calling
    # model.encode() once per chunk in a loop.
    texts = [c["text"] for c in chunks]

    # step 2: run the batch through the model.
    active_model = model if model is not None else get_model()
    embeddings = active_model.encode(texts)

    # step 3: attach each embedding back onto its corresponding chunk dict.
    # careful: model.encode() returns numpy arrays, not plain Python lists —
    # numpy arrays aren't JSON-serializable, which will matter once these
    # chunks get written to a vector DB in Phase 4. Convert with .tolist().
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    return chunks


def __getattr__(name: str) -> Any:
    """Support accessing embedder.model lazily without breaking backwards compatibility."""
    if name == "model":
        return get_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
