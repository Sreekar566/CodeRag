from unittest.mock import MagicMock, patch
import numpy as np

from src.storage.indexer import index_files, index_repository
from src.storage.memory_store import InMemoryVectorStore


def test_index_files_basic():
    store = InMemoryVectorStore(default_collection="test_idx")
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    files = [
        {
            "path": "app/routes.py",
            "content": "def index():\n    return 'Hello World'\n",
        }
    ]

    res = index_files(files, vector_store=store, collection_name="test_idx", embedder_model=mock_model)
    assert res["files_processed"] == 1
    assert res["chunks_stored"] == 1
    assert store.count("test_idx") == 1

    chunks = store.get(["app/routes.py:1-2"], collection_name="test_idx")
    assert len(chunks) == 1
    assert chunks[0]["path"] == "app/routes.py"
    assert chunks[0]["language"] == "python"


def test_index_files_empty():
    store = InMemoryVectorStore()
    res = index_files([], vector_store=store)
    assert res["files_processed"] == 0
    assert res["chunks_stored"] == 0


@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.filter_files")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_mocked(mock_get_content, mock_filter, mock_get_tree):
    mock_get_tree.return_value = [
        {"path": "app/auth.py", "type": "blob", "size": 100},
        {"path": "README.md", "type": "blob", "size": 200},
        {"path": "venv/bad.py", "type": "blob", "size": 50},
    ]

    mock_filter.return_value = [
        {"path": "app/auth.py", "type": "blob", "size": 100},
        {"path": "README.md", "type": "blob", "size": 200},
    ]

    def fake_content(owner, repo, path, token):
        if path == "app/auth.py":
            return "def login():\n    pass\n"
        elif path == "README.md":
            return "# Project Title\n"
        raise ValueError(f"Unknown path: {path}")

    mock_get_content.side_effect = fake_content

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    store = InMemoryVectorStore()

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        token="faketoken",
        vector_store=store,
        embedder_model=mock_model,
    )

    assert stats["owner"] == "testowner"
    assert stats["repo"] == "testrepo"
    assert stats["total_tree_files"] == 3
    assert stats["filtered_files"] == 2
    assert stats["chunks_stored"] == 2
    assert stats["skipped_files"] == []
    assert store.count("testowner_testrepo") == 2


@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.filter_files")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_handles_file_fetch_failure(mock_get_content, mock_filter, mock_get_tree):
    mock_get_tree.return_value = [
        {"path": "app/auth.py", "type": "blob", "size": 100},
        {"path": "corrupt.py", "type": "blob", "size": 100},
    ]
    mock_filter.return_value = [
        {"path": "app/auth.py", "type": "blob", "size": 100},
        {"path": "corrupt.py", "type": "blob", "size": 100},
    ]

    def fake_content(owner, repo, path, token):
        if path == "app/auth.py":
            return "def login(): pass"
        raise RuntimeError("API timeout fetching file")

    mock_get_content.side_effect = fake_content

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    store = InMemoryVectorStore()

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        vector_store=store,
        embedder_model=mock_model,
    )

    assert stats["filtered_files"] == 2
    assert stats["chunks_stored"] == 1
    assert "corrupt.py" in stats["skipped_files"]
    assert store.count("testowner_testrepo") == 1
