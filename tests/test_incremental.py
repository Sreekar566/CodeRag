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


def test_resolve_target_collection():
    indexer = IncrementalIndexer(vector_store=InMemoryVectorStore(), registry=RepoRegistry(storage_path=None))
    assert indexer._resolve_target_collection("owner", "repo", None) == "owner_repo"
    assert indexer._resolve_target_collection("owner", "repo", "custom/name") == "custom_name"


@patch("src.registry.incremental.get_default_branch")
def test_resolve_branch(mock_get_branch):
    indexer = IncrementalIndexer(vector_store=InMemoryVectorStore(), registry=RepoRegistry(storage_path=None))
    assert indexer._resolve_branch("owner", "repo", "token", "feature") == "feature"

    mock_get_branch.return_value = "develop"
    assert indexer._resolve_branch("owner", "repo", "token", "auto") == "develop"

    mock_get_branch.side_effect = Exception("Network error")
    assert indexer._resolve_branch("owner", "repo", "token", None) == "main"


def test_purge_chunks_for_files():
    store = InMemoryVectorStore()
    store.add_chunks(
        [
            {"id": "a.py:1-5", "path": "a.py", "text": "code a", "embedding": [0.1, 0.2]},
            {"id": "b.py:1-5", "path": "b.py", "text": "code b", "embedding": [0.3, 0.4]},
        ],
        collection_name="test_col",
    )
    indexer = IncrementalIndexer(vector_store=store, registry=RepoRegistry(storage_path=None))

    indexer._purge_chunks_for_files("test_col", ["a.py"])
    assert store.count("test_col") == 1
    assert len(store.get(["a.py:1-5"], collection_name="test_col")) == 0
    assert len(store.get(["b.py:1-5"], collection_name="test_col")) == 1


def test_determine_changes_and_cleanup_initial():
    store = InMemoryVectorStore()
    indexer = IncrementalIndexer(vector_store=store, registry=RepoRegistry(storage_path=None))

    items = [{"path": "file1.py", "sha": "s1"}, {"path": "file2.py", "sha": "s2"}]
    (
        to_fetch,
        added,
        modified,
        deleted,
        unchanged,
        is_initial,
        is_delta_unchanged,
    ) = indexer._determine_changes_and_cleanup("test_col", items, prev_meta=None, force=False)

    assert is_initial is True
    assert is_delta_unchanged is False
    assert to_fetch == ["file1.py", "file2.py"]
    assert added == ["file1.py", "file2.py"]
    assert modified == []
    assert deleted == []
    assert unchanged == 0


@patch("src.registry.incremental.get_file_content")
def test_fetch_file_contents_with_failures(mock_content):
    def fake_fetch(owner, repo, path, token):
        if path == "good.py":
            return "print('ok')"
        if path == "none.py":
            return None
        raise RuntimeError("Fetch failure")

    mock_content.side_effect = fake_fetch
    indexer = IncrementalIndexer(vector_store=InMemoryVectorStore(), registry=RepoRegistry(storage_path=None))

    fetched, skipped = indexer._fetch_file_contents("o", "r", "tok", ["good.py", "none.py", "bad.py"])
    assert len(fetched) == 1
    assert fetched[0] == {"path": "good.py", "content": "print('ok')"}
    assert skipped == ["none.py", "bad.py"]


@patch("src.registry.incremental.get_latest_commit_sha")
@patch("src.registry.incremental.get_repo_tree")
@patch("src.registry.incremental.audit_filter_files")
@patch("src.registry.incremental.get_file_content")
def test_incremental_indexer_failed_fetch_does_not_poison_shas_and_retries(
    mock_get_content, mock_audit_filter, mock_get_tree, mock_get_sha
):
    """
    Ensure transient fetch failures during delta updates neither purge existing chunks
    nor record unindexed SHAs / commit_sha, allowing subsequent runs to retry cleanly.
    """
    from src.observability.models import IngestionAuditLog

    store = InMemoryVectorStore()
    registry = RepoRegistry(storage_path=None)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    # Commit 1: initial index of a.py and b.py
    tree_c1 = [
        {"path": "a.py", "type": "blob", "sha": "sha_a1"},
        {"path": "b.py", "type": "blob", "sha": "sha_b1"},
    ]
    mock_get_sha.return_value = "commit_1"
    mock_get_tree.return_value = tree_c1
    mock_audit_filter.return_value = (
        tree_c1,
        IngestionAuditLog(repo_id="org/repo", branch="main", total_scanned=2, total_kept=2),
    )
    mock_get_content.side_effect = lambda o, r, p, t: f"# {p} v1\n"

    indexer = IncrementalIndexer(vector_store=store, registry=registry, embedder_model=mock_model)
    res1 = indexer.index_repository(owner="org", repo="repo", branch="main")
    assert res1.status == "created"
    assert store.count("org_repo") == 2

    # Commit 2: b.py modified, c.py added; fetching both fails transiently
    tree_c2 = [
        {"path": "a.py", "type": "blob", "sha": "sha_a1"},
        {"path": "b.py", "type": "blob", "sha": "sha_b2"},
        {"path": "c.py", "type": "blob", "sha": "sha_c1"},
    ]
    mock_get_sha.return_value = "commit_2"
    mock_get_tree.return_value = tree_c2
    mock_audit_filter.return_value = (
        tree_c2,
        IngestionAuditLog(repo_id="org/repo", branch="main", total_scanned=3, total_kept=3),
    )
    mock_get_content.side_effect = RuntimeError("Transient 502 error")

    res2 = indexer.index_repository(owner="org", repo="repo", branch="main")
    assert res2.files_processed == 0
    assert sorted(res2.skipped_files) == ["b.py", "c.py"]
    assert res2.commit_sha == "commit_1"

    # Existing chunks for b.py v1 must still be intact in the vector store
    b_chunks = store.query("b.py", top_k=5, collection_name="org_repo", where={"path": "b.py"})
    assert len(b_chunks) == 1
    assert "v1" in b_chunks[0]["text"]

    # Registry must NOT poison SHAs or advance commit_sha
    meta_after_fail = registry.get("org/repo")
    assert meta_after_fail.commit_sha == "commit_1"
    assert meta_after_fail.file_shas == {"a.py": "sha_a1", "b.py": "sha_b1"}
    assert "c.py" not in meta_after_fail.file_shas

    # Retry at commit_2 when GitHub recovers: must NOT short-circuit and must retry b.py & c.py only
    fetched_paths: list[str] = []

    def recover_content(o, r, p, t):
        fetched_paths.append(p)
        return f"# {p} v2\n"

    mock_get_content.side_effect = recover_content
    res3 = indexer.index_repository(owner="org", repo="repo", branch="main")

    assert res3.status == "updated"
    assert res3.commit_sha == "commit_2"
    assert sorted(fetched_paths) == ["b.py", "c.py"]
    assert res3.files_processed == 2
    assert store.count("org_repo") == 3

    b_chunks_v2 = store.query("b.py", top_k=5, collection_name="org_repo", where={"path": "b.py"})
    assert len(b_chunks_v2) == 1
    assert "v2" in b_chunks_v2[0]["text"]

    meta_recovered = registry.get("org/repo")
    assert meta_recovered.commit_sha == "commit_2"
    assert meta_recovered.file_shas == {
        "a.py": "sha_a1",
        "b.py": "sha_b2",
        "c.py": "sha_c1",
    }


@patch("src.registry.incremental.get_latest_commit_sha")
@patch("src.registry.incremental.get_repo_tree")
@patch("src.registry.incremental.audit_filter_files")
@patch("src.registry.incremental.get_file_content")
def test_incremental_indexer_initial_partial_failure_retries_only_failed(
    mock_get_content, mock_audit_filter, mock_get_tree, mock_get_sha
):
    """
    Ensure initial index with a transient file fetch failure leaves commit_sha None
    and omits only the failed file from file_shas so the next sync retries it.
    """
    from src.observability.models import IngestionAuditLog

    store = InMemoryVectorStore()
    registry = RepoRegistry(storage_path=None)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    tree = [
        {"path": "ok.py", "type": "blob", "sha": "sha_ok"},
        {"path": "flaky.py", "type": "blob", "sha": "sha_flaky"},
        {"path": "bin.dat", "type": "blob", "sha": "sha_bin"},
    ]
    mock_get_sha.return_value = "commit_init"
    mock_get_tree.return_value = tree
    mock_audit_filter.return_value = (
        tree,
        IngestionAuditLog(repo_id="org/proj", branch="main", total_scanned=3, total_kept=3),
    )

    def flaky_fetch(o, r, p, t):
        if p == "ok.py":
            return "ok = True\n"
        if p == "bin.dat":
            return None  # genuine binary file
        raise TimeoutError("Read timed out")

    mock_get_content.side_effect = flaky_fetch

    indexer = IncrementalIndexer(vector_store=store, registry=registry, embedder_model=mock_model)
    res1 = indexer.index_repository(owner="org", repo="proj", branch="main")

    assert res1.commit_sha is None
    assert sorted(res1.skipped_files) == ["bin.dat", "flaky.py"]
    meta1 = registry.get("org/proj")
    assert meta1.commit_sha is None
    # ok.py and binary bin.dat are recorded; flaky.py is omitted for retry
    assert meta1.file_shas == {"ok.py": "sha_ok", "bin.dat": "sha_bin"}

    # Retry: only flaky.py should be fetched
    retried_paths: list[str] = []

    def fixed_fetch(o, r, p, t):
        retried_paths.append(p)
        return "flaky = False\n"

    mock_get_content.side_effect = fixed_fetch
    res2 = indexer.index_repository(owner="org", repo="proj", branch="main")

    assert retried_paths == ["flaky.py"]
    assert res2.commit_sha == "commit_init"
    meta2 = registry.get("org/proj")
    assert meta2.commit_sha == "commit_init"
    assert meta2.file_shas == {"ok.py": "sha_ok", "bin.dat": "sha_bin", "flaky.py": "sha_flaky"}



