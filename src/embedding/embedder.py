# Code-specialized embedding model (not a general-purpose text model) —
# see project_log.md for the reasoning: retrieval quality for code questions
# should benefit from a model actually trained on code.
# Originally jinaai/jina-embeddings-v2-base-code, but its trust_remote_code
# custom loading path deadlocked on this machine with no diagnosable error
# (see project_log.md) — switched to a standard-architecture code model
# that sentence-transformers loads natively, no custom remote code needed.
MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"

# Loaded once at import time, not inside embed_chunks() — loading the model
# is expensive (downloads weights the first time, then loads them into
# memory), so you don't want to pay that cost on every call.
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Takes the chunk dicts produced by chunk_text() and adds an "embedding"
    vector to each one, in place.

    This is the swappable interface: any other embedding backend (a hosted
    API, a different local model) should implement the same signature —
    take chunks in, return the same chunks with "embedding" added.
    """
    # step 1: pull out just the text from each chunk into a single list —
    # encoding all chunks in one batched call is much faster than calling
    # model.encode() once per chunk in a loop.
    # hint: texts = [c["text"] for c in chunks]

    texts = []
    for c in chunks:
        texts.append(c["text"])




    # step 2: run the batch through the model.
    # hint: embeddings = model.encode(texts)

    embeddings = model.encode(texts)

    # step 3: attach each embedding back onto its corresponding chunk dict.
    # careful: model.encode() returns numpy arrays, not plain Python lists —
    # numpy arrays aren't JSON-serializable, which will matter once these
    # chunks get written to a vector database. Convert with .tolist().
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks