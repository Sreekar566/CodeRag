from unittest.mock import MagicMock
import pytest

from src.storage.base import sanitize_collection_name
from src.storage.chroma_store import ChromaVectorStore, get_vector_store
from src.storage.memory_store import InMemoryVectorStore


def test_sanitize_collection_name():
    assert sanitize_collection_name("miguelgrinberg/microblog") == "miguelgrinberg_microblog"
    assert sanitize_collection_name("valid_name-123") == "valid_name-123"
    assert sanitize_collection_name("a/b") == "a_b"
    assert sanitize_collection_name("---") == "col"
    assert sanitize_collection_name("...test...") == "test"
    # Over 63 chars
    long_name = "a" * 100
    sanitized = sanitize_collection_name(long_name)
    assert len(sanitized) <= 63
    assert sanitized == "a" * 63


def test_chroma_store_basic_operations():
    # Use ephemeral in-memory chroma client (persist_directory=None)
    store = ChromaVectorStore(persist_directory=None, default_collection="test_basic")
    store.clear("test_basic")

    assert store.count("test_basic") == 0
    assert store.query("empty", collection_name="test_basic") == []

    # Add chunks with precomputed embeddings
    chunks = [
        {
            "id": "app/auth.py:1-20",
            "path": "app/auth.py",
            "start_line": 1,
            "end_line": 20,
            "text": "def login(): pass",
            "language": "python",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "app/models.py:1-30",
            "path": "app/models.py",
            "start_line": 1,
            "end_line": 30,
            "text": "class User(db.Model): pass",
            "language": "python",
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "id": "src/index.ts:1-15",
            "path": "src/index.ts",
            "start_line": 1,
            "end_line": 15,
            "text": "console.log('hello');",
            "language": "typescript",
            "embedding": [0.0, 0.0, 1.0],
        },
    ]

    added = store.add_chunks(chunks, collection_name="test_basic")
    assert added == 3
    assert store.count("test_basic") == 3

    # Test get by ID
    retrieved = store.get(["app/auth.py:1-20"], collection_name="test_basic")
    assert len(retrieved) == 1
    assert retrieved[0]["id"] == "app/auth.py:1-20"
    assert retrieved[0]["path"] == "app/auth.py"
    assert retrieved[0]["language"] == "python"
    assert retrieved[0]["text"] == "def login(): pass"

    # Test query by embedding closest to auth chunk
    results = store.query_by_embedding([0.9, 0.1, 0.0], top_k=2, collection_name="test_basic")
    assert len(results) == 2
    assert results[0]["id"] == "app/auth.py:1-20"
    assert results[0]["similarity"] > 0.9

    # Test query with custom embed_fn
    mock_embed = MagicMock(return_value=[0.0, 0.9, 0.1])
    store_with_embed = ChromaVectorStore(
        persist_directory=None,
        default_collection="test_basic",
        embed_fn=mock_embed,
        client=store._client,
    )
    query_results = store_with_embed.query("find user model", top_k=1, collection_name="test_basic")
    assert len(query_results) == 1
    assert query_results[0]["id"] == "app/models.py:1-30"

    # Test where filtering
    ts_results = store.query_by_embedding(
        [0.0, 0.0, 1.0],
        top_k=5,
        collection_name="test_basic",
        where={"language": "typescript"},
    )
    assert len(ts_results) == 1
    assert ts_results[0]["language"] == "typescript"

    # Test delete
    store.delete(ids=["src/index.ts:1-15"], collection_name="test_basic")
    assert store.count("test_basic") == 2

    # Test clear
    store.clear("test_basic")
    assert store.count("test_basic") == 0


def test_chroma_store_auto_id_and_auto_embedding():
    mock_embed = MagicMock(return_value=[0.5, 0.5])
    store = ChromaVectorStore(persist_directory=None, default_collection="test_auto", embed_fn=mock_embed)

    # Chunks without ID or embedding
    chunks = [
        {
            "path": "test/sample.py",
            "start_line": 5,
            "end_line": 15,
            "text": "x = 42",
            "language": "python",
        }
    ]

    added = store.add_chunks(chunks, collection_name="test_auto")
    assert added == 1
    mock_embed.assert_called_once_with("x = 42")

    items = store.get(["test/sample.py:5-15"], collection_name="test_auto")
    assert len(items) == 1
    assert items[0]["start_line"] == 5
    assert items[0]["end_line"] == 15


def test_in_memory_vector_store():
    store = InMemoryVectorStore(default_collection="test_mem")
    assert store.count("test_mem") == 0
    assert store.query("empty", collection_name="test_mem") == []

    chunks = [
        {
            "id": "c1",
            "path": "auth.py",
            "start_line": 1,
            "end_line": 10,
            "text": "login function",
            "language": "python",
            "embedding": [1.0, 0.0],
        },
        {
            "id": "c2",
            "path": "search.py",
            "start_line": 1,
            "end_line": 10,
            "text": "search function",
            "language": "python",
            "embedding": [0.0, 1.0],
        },
    ]
    added = store.add_chunks(chunks, collection_name="test_mem")
    assert added == 2
    assert store.count("test_mem") == 2

    # Query closest to c2
    results = store.query_by_embedding([0.1, 0.9], top_k=1, collection_name="test_mem")
    assert len(results) == 1
    assert results[0]["id"] == "c2"

    # Get
    item = store.get(["c1"], collection_name="test_mem")
    assert len(item) == 1
    assert item[0]["id"] == "c1"

    # Delete
    store.delete(ids=["c1"], collection_name="test_mem")
    assert store.count("test_mem") == 1

    # Clear
    store.clear("test_mem")
    assert store.count("test_mem") == 0


def test_factory_get_vector_store():
    store = get_vector_store(persist_directory=None, default_collection="factory_test")
    assert isinstance(store, ChromaVectorStore)
    assert store.default_collection == "factory_test"
