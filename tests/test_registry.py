"""
Unit tests for RepositoryMetadata and RepoRegistry.
Covers metadata serialization, thread-safe registry operations, persistence, and concurrency locks.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import pytest

from src.registry.models import RepositoryMetadata, ChangeSet, IncrementalIndexResult
from src.registry.registry import RepoRegistry, get_repo_registry


def test_repository_metadata_serialization():
    meta = RepositoryMetadata(
        owner="test-owner",
        repo="test-repo",
        collection_name="test_owner_test_repo",
        branch="main",
        commit_sha="a1b2c3d4e5",
        indexed_at="2026-10-20T12:00:00Z",
        file_count=10,
        chunk_count=35,
        file_shas={"app/main.py": "sha123", "README.md": "sha456"},
        status="ready",
        metadata={"language": "python"},
    )

    assert meta.repo_id == "test-owner/test-repo"
    d = meta.to_dict()
    assert d["owner"] == "test-owner"
    assert d["repo"] == "test-repo"
    assert d["repo_id"] == "test-owner/test-repo"
    assert d["commit_sha"] == "a1b2c3d4e5"
    assert d["file_shas"]["app/main.py"] == "sha123"

    restored = RepositoryMetadata.from_dict(d)
    assert restored.owner == meta.owner
    assert restored.repo == meta.repo
    assert restored.repo_id == meta.repo_id
    assert restored.commit_sha == meta.commit_sha
    assert restored.file_shas == meta.file_shas
    assert restored.file_count == 10
    assert restored.chunk_count == 35


def test_changeset_properties():
    cs = ChangeSet(
        added_files=["new.py"],
        modified_files=["mod.py"],
        deleted_files=["del.py"],
        unchanged_files=["keep.py"],
    )
    assert cs.has_changes is True
    assert cs.total_files == 4

    cs_empty = ChangeSet(unchanged_files=["a.py", "b.py"])
    assert cs_empty.has_changes is False
    assert cs_empty.total_files == 2


def test_incremental_index_result_serialization():
    res = IncrementalIndexResult(
        owner="owner",
        repo="repo",
        collection="owner_repo",
        branch="main",
        commit_sha="c0ffee",
        status="updated",
        files_processed=3,
        chunks_stored=12,
        files_added=1,
        files_modified=2,
        files_deleted=1,
        files_unchanged=5,
        chunks_added=12,
        chunks_deleted=8,
        total_tree_files=10,
        filtered_files=8,
    )
    d = res.to_dict()
    assert d["status"] == "updated"
    assert d["files_modified"] == 2
    assert d["chunks_added"] == 12


def test_repo_registry_in_memory():
    reg = RepoRegistry(storage_path=None)

    meta1 = RepositoryMetadata(owner="alice", repo="project-a", collection_name="alice_project_a")
    meta2 = RepositoryMetadata(owner="bob", repo="project-b", collection_name="bob_project_b")

    reg.register(meta1)
    reg.register(meta2)

    assert reg.exists("alice", "project-a")
    assert reg.exists("alice/project-a")
    assert reg.exists("BOB/PROJECT-B")  # case-insensitive normalization
    assert not reg.exists("charlie", "project-c")

    fetched = reg.get("alice", "project-a")
    assert fetched is not None
    assert fetched.owner == "alice"
    assert fetched.collection_name == "alice_project_a"

    all_repos = reg.list_repos()
    assert len(all_repos) == 2

    # Update status
    reg.update_status("alice", "project-a", status="indexing", chunk_count=50)
    updated = reg.get("alice/project-a")
    assert updated.status == "indexing"
    assert updated.chunk_count == 50

    # Delete
    deleted = reg.delete("alice", "project-a")
    assert deleted is True
    assert not reg.exists("alice/project-a")
    assert len(reg.list_repos()) == 1


def test_repo_registry_disk_persistence(tmp_path):
    storage_file = str(tmp_path / "repos.json")
    reg1 = RepoRegistry(storage_path=storage_file)

    meta = RepositoryMetadata(
        owner="org",
        repo="lib",
        collection_name="org_lib",
        branch="develop",
        commit_sha="fedcba9876",
        file_count=5,
        chunk_count=20,
        file_shas={"lib/core.py": "hash1"},
    )
    reg1.register(meta)

    assert os.path.exists(storage_file)
    with open(storage_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "org/lib" in data["repositories"]

    # Reload in a second registry instance pointing to the same file
    reg2 = RepoRegistry(storage_path=storage_file)
    loaded = reg2.get("org/lib")
    assert loaded is not None
    assert loaded.commit_sha == "fedcba9876"
    assert loaded.file_shas == {"lib/core.py": "hash1"}
    assert loaded.branch == "develop"


def test_repo_locking():
    reg = RepoRegistry(storage_path=None)

    # First acquisition succeeds
    assert reg.acquire_repo_lock("owner", "repo", blocking=True) is True

    # Second acquisition non-blocking fails because already locked
    assert reg.acquire_repo_lock("owner", "repo", blocking=False) is False

    # Different repo lock succeeds
    assert reg.acquire_repo_lock("other", "repo", blocking=False) is True
    reg.release_repo_lock("other", "repo")

    # Release original lock
    reg.release_repo_lock("owner", "repo")

    # Now can be acquired again
    assert reg.acquire_repo_lock("owner", "repo", blocking=False) is True
    reg.release_repo_lock("owner", "repo")


def test_concurrent_registry_writes(tmp_path):
    storage_file = str(tmp_path / "concurrent_reg.json")
    reg = RepoRegistry(storage_path=storage_file)

    def register_worker(worker_id: int):
        for i in range(10):
            meta = RepositoryMetadata(
                owner=f"worker_{worker_id}",
                repo=f"repo_{i}",
                collection_name=f"worker_{worker_id}_repo_{i}",
                file_count=i,
            )
            reg.register(meta)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(register_worker, wid) for wid in range(5)]
        concurrent.futures.wait(futures)

    # 5 workers * 10 repos = 50 total repositories registered safely without corruption
    assert len(reg.list_repos()) == 50

    # Ensure persisted file is valid JSON
    with open(storage_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["repositories"]) == 50
