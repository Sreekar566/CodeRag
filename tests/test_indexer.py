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


@patch("src.storage.indexer.get_vector_store")
def test_index_files_default_vector_store(mock_get_store):
    fake_store = InMemoryVectorStore(default_collection="codebase")
    mock_get_store.return_value = fake_store

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    files = [{"path": "main.py", "content": "print('hello')\n"}]
    res = index_files(files, embedder_model=mock_model)

    mock_get_store.assert_called_once()
    assert res["files_processed"] == 1
    assert res["chunks_stored"] == 1
    assert fake_store.count("codebase") == 1


def test_index_files_skips_empty_path_or_content():
    store = InMemoryVectorStore(default_collection="test_empty")
    mock_model = MagicMock()

    files = [
        {"path": "", "content": "def foo(): pass"},
        {"path": "empty.py", "content": ""},
        {"path": "", "content": ""},
    ]
    res = index_files(files, vector_store=store, embedder_model=mock_model)
    assert res["files_processed"] == 3
    assert res["chunks_stored"] == 0
    assert store.count("test_empty") == 0
    mock_model.encode.assert_not_called()


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


@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.filter_files")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_handles_binary_files_as_skipped(mock_get_content, mock_filter, mock_get_tree):
    mock_get_tree.return_value = [
        {"path": "app/auth.py", "type": "blob", "size": 100},
        {"path": "binary.dat", "type": "blob", "size": 100},
    ]
    mock_filter.return_value = [
        {"path": "app/auth.py", "type": "blob", "size": 100},
        {"path": "binary.dat", "type": "blob", "size": 100},
    ]

    def fake_content(owner, repo, path, token):
        if path == "app/auth.py":
            return "def login(): pass\n"
        elif path == "binary.dat":
            return None  # binary / undecodable file returns None
        raise RuntimeError(f"Unexpected path: {path}")

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
    assert stats["files_processed"] == 1
    assert stats["chunks_stored"] == 1
    assert stats["skipped_files"] == ["binary.dat"]
    assert store.count("testowner_testrepo") == 1


@patch("src.storage.indexer.get_vector_store")
@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_default_vector_store(
    mock_get_content, mock_get_tree, mock_get_store
):
    fake_store = InMemoryVectorStore()
    mock_get_store.return_value = fake_store

    mock_get_tree.return_value = [{"path": "app.py", "type": "blob", "size": 100}]
    mock_get_content.return_value = "print('hello')\n"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        embedder_model=mock_model,
    )

    mock_get_store.assert_called_once()
    assert stats["owner"] == "testowner"
    assert stats["repo"] == "testrepo"
    assert stats["chunks_stored"] == 1
    assert fake_store.count("testowner_testrepo") == 1


@patch("src.storage.indexer.get_default_branch")
@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_auto_branch_detection_success(
    mock_get_content, mock_get_tree, mock_get_default_branch
):
    mock_get_default_branch.return_value = "develop"
    mock_get_tree.return_value = [{"path": "app.py", "type": "blob", "size": 100}]
    mock_get_content.return_value = "def run(): pass\n"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    store = InMemoryVectorStore()

    # Test branch="auto"
    stats_auto = index_repository(
        owner="testowner",
        repo="testrepo",
        token="tok_123",
        branch="auto",
        vector_store=store,
        embedder_model=mock_model,
    )
    mock_get_default_branch.assert_called_with("testowner", "testrepo", "tok_123")
    mock_get_tree.assert_called_with("testowner", "testrepo", "tok_123", branch="develop")
    assert stats_auto["branch"] == "develop"

    # Test branch=None also auto-detects
    mock_get_default_branch.reset_mock()
    stats_none = index_repository(
        owner="testowner",
        repo="testrepo",
        token="tok_123",
        branch=None,
        vector_store=store,
        embedder_model=mock_model,
    )
    mock_get_default_branch.assert_called_with("testowner", "testrepo", "tok_123")
    assert stats_none["branch"] == "develop"


@patch("src.storage.indexer.get_default_branch")
@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_auto_branch_fallback_on_exception(
    mock_get_content, mock_get_tree, mock_get_default_branch
):
    mock_get_default_branch.side_effect = RuntimeError("API timeout fetching default branch")
    mock_get_tree.return_value = [{"path": "app.py", "type": "blob", "size": 100}]
    mock_get_content.return_value = "def run(): pass\n"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    store = InMemoryVectorStore()

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        branch="auto",
        vector_store=store,
        embedder_model=mock_model,
    )
    mock_get_tree.assert_called_with("testowner", "testrepo", None, branch="main")
    assert stats["branch"] == "main"


@patch("src.registry.incremental.index_repository_incremental")
def test_index_repository_incremental_delegation(mock_index_incremental):
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {
        "owner": "testowner",
        "repo": "testrepo",
        "collection": "custom_col",
        "branch": "feature/v2",
        "status": "ready",
        "chunks_stored": 12,
    }
    mock_index_incremental.return_value = mock_result

    store = InMemoryVectorStore()
    mock_registry = MagicMock()
    mock_model = MagicMock()

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        token="gh_token",
        branch="feature/v2",
        collection_name="custom_col",
        incremental=True,
        force=True,
        vector_store=store,
        registry=mock_registry,
        embedder_model=mock_model,
    )

    mock_index_incremental.assert_called_once_with(
        owner="testowner",
        repo="testrepo",
        token="gh_token",
        branch="feature/v2",
        collection_name="custom_col",
        force=True,
        vector_store=store,
        registry=mock_registry,
        embedder_model=mock_model,
    )
    assert stats["status"] == "ready"
    assert stats["chunks_stored"] == 12
    assert stats["collection"] == "custom_col"


@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_registers_in_registry(
    mock_get_content, mock_get_tree
):
    mock_get_tree.return_value = [
        {"path": "app/main.py", "type": "blob", "size": 100, "sha": "sha_main"},
        {"path": "README.md", "type": "blob", "size": 50, "sha": "sha_readme"},
    ]
    mock_get_content.side_effect = lambda owner, repo, path, token: "content of " + path

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    store = InMemoryVectorStore()
    mock_registry = MagicMock()

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        branch="main",
        vector_store=store,
        registry=mock_registry,
        embedder_model=mock_model,
    )

    assert stats["chunks_stored"] == 2
    mock_registry.register.assert_called_once()
    meta = mock_registry.register.call_args[0][0]
    assert meta.owner == "testowner"
    assert meta.repo == "testrepo"
    assert meta.collection_name == "testowner_testrepo"
    assert meta.branch == "main"
    assert meta.status == "ready"
    assert meta.file_count == 2
    assert meta.chunk_count == 2
    assert meta.file_shas == {
        "app/main.py": "sha_main",
        "README.md": "sha_readme",
    }
    assert "audit_log" in meta.metadata
    assert meta.indexed_at is not None


@patch("src.storage.indexer.audit_filter_files")
@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_skips_items_without_path(
    mock_get_content, mock_get_tree, mock_audit_filter
):
    mock_get_tree.return_value = [
        {"path": "valid.py", "type": "blob", "size": 50},
        {"path": "", "type": "blob", "size": 50},
        {"type": "blob", "size": 50},
    ]
    mock_audit_log = MagicMock()
    mock_audit_log.to_dict.return_value = {"total": 3}
    mock_audit_filter.return_value = (
        [
            {"path": "valid.py", "type": "blob", "size": 50},
            {"path": "", "type": "blob", "size": 50},
            {"type": "blob", "size": 50},
        ],
        mock_audit_log,
    )
    mock_get_content.return_value = "valid = True\n"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    store = InMemoryVectorStore()
    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        vector_store=store,
        embedder_model=mock_model,
    )

    mock_get_content.assert_called_once_with("testowner", "testrepo", "valid.py", None)
    assert stats["filtered_files"] == 3
    assert stats["chunks_stored"] == 1
    assert stats["skipped_files"] == []


@patch("src.storage.indexer.get_repo_tree")
@patch("src.storage.indexer.get_file_content")
def test_index_repository_custom_collection_name(
    mock_get_content, mock_get_tree
):
    mock_get_tree.return_value = [{"path": "hello.py", "type": "blob", "size": 20}]
    mock_get_content.return_value = "print('world')\n"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    store = InMemoryVectorStore()

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        collection_name="Custom/Collection:Name",
        vector_store=store,
        embedder_model=mock_model,
    )

    assert stats["collection"] == "Custom_Collection_Name"
    assert store.count("Custom_Collection_Name") == 1
