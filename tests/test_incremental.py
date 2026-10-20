"""
Unit tests for incremental repository indexing and change detection.
Mocked GitHub API, embedder, and vector store ensure fast, offline execution.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.registry.incremental import detect_changes, IncrementalIndexer, index_repository_incremental
from src.registry.registry import RepoRegistry
from src.storage.indexer import index_repository
from src.storage.memory_store import InMemoryVectorStore


def test_detect_changes_matrix():
    current_tree = [
        {"path": "app/auth.py", "sha": "sha_auth_v2"},  # Modified
        {"path": "app/models.py", "sha": "sha_models_v1"},  # Unchanged
        {"path": "app/billing.py", "sha": "sha_billing_v1"},  # Added
    ]

    prev_shas = {
        "app/auth.py": "sha_auth_v1",
        "app/models.py": "sha_models_v1",
        "app/old_feature.py": "sha_old_v1",  # Deleted
    }

    changes = detect_changes(current_tree, prev_shas)

    assert changes.has_changes is True
    assert changes.added_files == ["app/billing.py"]
    assert changes.modified_files == ["app/auth.py"]
    assert changes.deleted_files == ["app/old_feature.py"]
    assert changes.unchanged_files == ["app/models.py"]
    assert changes.total_files == 4


def test_detect_changes_no_changes():
    current_tree = [
        {"path": "a.py", "sha": "sha1"},
        {"path": "b.py", "sha": "sha2"},
    ]
    prev_shas = {"a.py": "sha1", "b.py": "sha2"}

    changes = detect_changes(current_tree, prev_shas)
    assert changes.has_changes is False
    assert changes.unchanged_files == ["a.py", "b.py"]
    assert changes.added_files == []
    assert changes.modified_files == []
    assert changes.deleted_files == []


@patch("src.registry.incremental.get_latest_commit_sha")
@patch("src.registry.incremental.get_repo_tree")
@patch("src.registry.incremental.filter_files")
@patch("src.registry.incremental.get_file_content")
def test_incremental_indexer_initial_run(
    mock_get_content, mock_filter, mock_get_tree, mock_get_sha
):
    mock_get_sha.return_value = "commit_111"
    mock_get_tree.return_value = [
        {"path": "app/routes.py", "type": "blob", "sha": "sha_r1"},
        {"path": "README.md", "type": "blob", "sha": "sha_rm1"},
    ]
    mock_filter.return_value = [
        {"path": "app/routes.py", "type": "blob", "sha": "sha_r1"},
        {"path": "README.md", "type": "blob", "sha": "sha_rm1"},
    ]
    mock_get_content.side_effect = lambda o, r, p, t: f"# content for {p}"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    store = InMemoryVectorStore()
    registry = RepoRegistry(storage_path=None)

    indexer = IncrementalIndexer(
        vector_store=store,
        registry=registry,
        embedder_model=mock_model,
    )

    res = indexer.index_repository(
        owner="myorg",
        repo="myrepo",
        branch="main",
    )

    assert res.status == "created"
    assert res.commit_sha == "commit_111"
    assert res.files_added == 2
    assert res.files_modified == 0
    assert res.files_deleted == 0
    assert res.files_unchanged == 0
    assert res.chunks_stored == 2
    assert store.count("myorg_myrepo") == 2

    # Verify registered metadata
    meta = registry.get("myorg/myrepo")
    assert meta is not None
    assert meta.commit_sha == "commit_111"
    assert meta.file_count == 2
    assert meta.file_shas == {
        "app/routes.py": "sha_r1",
        "README.md": "sha_rm1",
    }


@patch("src.registry.incremental.get_latest_commit_sha")
@patch("src.registry.incremental.get_repo_tree")
def test_incremental_indexer_unchanged_fast_path(
    mock_get_tree, mock_get_sha
):
    # Setup already indexed repo in registry
    registry = RepoRegistry(storage_path=None)
    from src.registry.models import RepositoryMetadata
    registry.register(
        RepositoryMetadata(
            owner="myorg",
            repo="myrepo",
            collection_name="myorg_myrepo",
            branch="main",
            commit_sha="commit_111",
            file_count=2,
            chunk_count=2,
            file_shas={"app/routes.py": "sha_r1", "README.md": "sha_rm1"},
            status="ready",
        )
    )

    # Remote commit SHA is unchanged
    mock_get_sha.return_value = "commit_111"

    store = InMemoryVectorStore()
    indexer = IncrementalIndexer(vector_store=store, registry=registry)

    res = indexer.index_repository(
        owner="myorg",
        repo="myrepo",
        branch="main",
    )

    assert res.status == "unchanged"
    assert res.commit_sha == "commit_111"
    assert res.files_processed == 0
    assert res.chunks_stored == 0
    # Tree should NOT even be fetched when commit SHA matches
    mock_get_tree.assert_not_called()


@patch("src.registry.incremental.get_latest_commit_sha")
@patch("src.registry.incremental.get_repo_tree")
@patch("src.registry.incremental.filter_files")
@patch("src.registry.incremental.get_file_content")
def test_incremental_indexer_delta_updates(
    mock_get_content, mock_filter, mock_get_tree, mock_get_sha
):
    # Initial repository state in store & registry
    store = InMemoryVectorStore()
    registry = RepoRegistry(storage_path=None)

    # Pre-populate store with chunks for 3 files: routes, auth, old_file
    initial_chunks = [
        {"id": "app/routes.py:1-10", "path": "app/routes.py", "text": "routes v1", "embedding": [0.1, 0.2]},
        {"id": "app/auth.py:1-10", "path": "app/auth.py", "text": "auth v1", "embedding": [0.3, 0.4]},
        {"id": "app/old_file.py:1-10", "path": "app/old_file.py", "text": "old v1", "embedding": [0.5, 0.6]},
    ]
    store.add_chunks(initial_chunks, collection_name="myorg_myrepo")
    assert store.count("myorg_myrepo") == 3

    from src.registry.models import RepositoryMetadata
    registry.register(
        RepositoryMetadata(
            owner="myorg",
            repo="myrepo",
            collection_name="myorg_myrepo",
            branch="main",
            commit_sha="commit_100",
            file_count=3,
            chunk_count=3,
            file_shas={
                "app/routes.py": "sha_r1",
                "app/auth.py": "sha_a1",
                "app/old_file.py": "sha_old",
            },
            status="ready",
        )
    )

    # Next commit:
    # app/routes.py unchanged (sha_r1)
    # app/auth.py modified (sha_a2)
    # app/old_file.py deleted (missing from tree)
    # app/new_file.py added (sha_n1)
    mock_get_sha.return_value = "commit_200"
    tree = [
        {"path": "app/routes.py", "type": "blob", "sha": "sha_r1"},
        {"path": "app/auth.py", "type": "blob", "sha": "sha_a2"},
        {"path": "app/new_file.py", "type": "blob", "sha": "sha_n1"},
    ]
    mock_get_tree.return_value = tree
    mock_filter.return_value = tree

    def fake_content(owner, repo, path, token):
        if path == "app/auth.py":
            return "auth v2 modified content\n"
        if path == "app/new_file.py":
            return "new file content\n"
        raise AssertionError(f"Unexpected fetch for path: {path}")

    mock_get_content.side_effect = fake_content

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.9, 0.9]])

    indexer = IncrementalIndexer(
        vector_store=store,
        registry=registry,
        embedder_model=mock_model,
    )

    res = indexer.index_repository(
        owner="myorg",
        repo="myrepo",
        branch="main",
    )

    assert res.status == "updated"
    assert res.commit_sha == "commit_200"
    assert res.files_added == 1
    assert res.files_modified == 1
    assert res.files_deleted == 1
    assert res.files_unchanged == 1
    assert res.files_processed == 2  # Only auth and new_file fetched!

    # Old file chunks should have been deleted from store
    old_chunks = store.get(["app/old_file.py:1-10"], collection_name="myorg_myrepo")
    assert len(old_chunks) == 0

    # Unchanged routes file chunk should still exist
    routes_chunks = store.get(["app/routes.py:1-10"], collection_name="myorg_myrepo")
    assert len(routes_chunks) == 1
    assert routes_chunks[0]["text"] == "routes v1"

    # Updated auth chunk should exist
    auth_query = store.query("auth", top_k=5, collection_name="myorg_myrepo", where={"path": "app/auth.py"})
    assert len(auth_query) >= 1
    assert "auth v2" in auth_query[0]["text"]

    # Total items in store: routes (1) + auth (1) + new_file (1) = 3
    assert store.count("myorg_myrepo") == 3

    # Registry updated with commit_200 and new file SHAs
    meta = registry.get("myorg/myrepo")
    assert meta.commit_sha == "commit_200"
    assert "app/old_file.py" not in meta.file_shas
    assert meta.file_shas["app/auth.py"] == "sha_a2"
    assert meta.file_shas["app/new_file.py"] == "sha_n1"


@patch("src.registry.incremental.get_default_branch")
@patch("src.registry.incremental.get_latest_commit_sha")
@patch("src.registry.incremental.get_repo_tree")
@patch("src.registry.incremental.filter_files")
@patch("src.registry.incremental.get_file_content")
def test_incremental_auto_branch_detection(
    mock_get_content, mock_filter, mock_get_tree, mock_get_sha, mock_get_branch
):
    mock_get_branch.return_value = "trunk"
    mock_get_sha.return_value = "trunk_sha_999"
    mock_get_tree.return_value = [{"path": "main.py", "type": "blob", "sha": "s1"}]
    mock_filter.return_value = [{"path": "main.py", "type": "blob", "sha": "s1"}]
    mock_get_content.return_value = "print('hello')"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    store = InMemoryVectorStore()
    registry = RepoRegistry(storage_path=None)

    res = index_repository_incremental(
        owner="org",
        repo="project",
        branch="auto",
        vector_store=store,
        registry=registry,
        embedder_model=mock_model,
    )

    assert res.branch == "trunk"
    mock_get_branch.assert_called_once_with("org", "project", None)
    mock_get_sha.assert_called_once_with("org", "project", None, branch="trunk")


@patch("src.registry.incremental.get_latest_commit_sha")
@patch("src.registry.incremental.get_repo_tree")
@patch("src.registry.incremental.filter_files")
@patch("src.registry.incremental.get_file_content")
def test_index_repository_with_incremental_flag(
    mock_get_content, mock_filter, mock_get_tree, mock_get_sha
):
    mock_get_sha.return_value = "sha_abc"
    mock_get_tree.return_value = [{"path": "main.py", "type": "blob", "sha": "s1"}]
    mock_filter.return_value = [{"path": "main.py", "type": "blob", "sha": "s1"}]
    mock_get_content.return_value = "print('hello')"

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    store = InMemoryVectorStore()
    registry = RepoRegistry(storage_path=None)

    stats = index_repository(
        owner="testowner",
        repo="testrepo",
        vector_store=store,
        embedder_model=mock_model,
        incremental=True,
        registry=registry,
    )

    assert stats["owner"] == "testowner"
    assert stats["repo"] == "testrepo"
    assert stats["status"] == "created"
    assert stats["commit_sha"] == "sha_abc"
    assert stats["chunks_stored"] == 1
    assert registry.exists("testowner/testrepo")


@pytest.mark.integration
def test_incremental_indexing_chroma_integration(tmp_path):
    """
    Integration test verifying incremental indexing with real Chroma persistent storage.
    """
    import shutil
    from src.storage.chroma_store import ChromaVectorStore

    chroma_dir = str(tmp_path / "chroma_inc")
    registry_file = str(tmp_path / "registry.json")

    store = ChromaVectorStore(persist_directory=chroma_dir, default_collection="inc_test")
    registry = RepoRegistry(storage_path=registry_file)

    with patch("src.registry.incremental.get_latest_commit_sha") as mock_sha, \
         patch("src.registry.incremental.get_repo_tree") as mock_tree, \
         patch("src.registry.incremental.filter_files") as mock_filter, \
         patch("src.registry.incremental.get_file_content") as mock_content:

        # Commit 1
        mock_sha.return_value = "c1"
        mock_tree.return_value = [
            {"path": "service.py", "type": "blob", "sha": "s_v1"},
            {"path": "utils.py", "type": "blob", "sha": "u_v1"},
        ]
        mock_filter.return_value = mock_tree.return_value
        mock_content.side_effect = lambda o, r, p, t: f"def {p.split('.')[0]}():\n    return '{p}'\n"

        indexer = IncrementalIndexer(vector_store=store, registry=registry)
        res1 = indexer.index_repository(owner="realorg", repo="realrepo", branch="main")

        assert res1.status == "created"
        assert res1.files_added == 2
        assert store.count("realorg_realrepo") == 2

        # Commit 2: update utils.py, remove service.py, add new.py
        mock_sha.return_value = "c2"
        mock_tree.return_value = [
            {"path": "utils.py", "type": "blob", "sha": "u_v2"},
            {"path": "new.py", "type": "blob", "sha": "n_v1"},
        ]
        mock_filter.return_value = mock_tree.return_value

        res2 = indexer.index_repository(owner="realorg", repo="realrepo", branch="main")

        assert res2.status == "updated"
        assert res2.files_added == 1
        assert res2.files_modified == 1
        assert res2.files_deleted == 1
        assert store.count("realorg_realrepo") == 2

    shutil.rmtree(chroma_dir, ignore_errors=True)

