"""
Thread-safe repository registry storing indexing state, commit SHAs, and file metadata.
Supports persistent JSON storage with atomic writes or in-memory operation for tests.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Any

from src.registry.models import RepositoryMetadata

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "./data/registry.json"


def _normalize_repo_id(owner_or_id: str, repo: str | None = None) -> str:
    """Normalize owner and repo inputs to 'owner/repo' lowercase."""
    if repo is not None:
        return f"{owner_or_id.strip()}/{repo.strip()}".lower()
    val = owner_or_id.strip().lower()
    return val


class RepoRegistry:
    """
    Manages metadata for indexed codebases.
    Ensures thread safety with reentrant locks and crash resilience with atomic writes.
    """

    def __init__(self, storage_path: str | None = DEFAULT_REGISTRY_PATH):
        self.storage_path = storage_path
        self._lock = threading.RLock()
        self._repo_locks: dict[str, threading.Lock] = {}
        self._repo_locks_guard = threading.Lock()
        self._repos: dict[str, RepositoryMetadata] = {}

        if self.storage_path:
            self._load()

    def _load(self) -> None:
        """Load registry from persistent JSON file if it exists."""
        with self._lock:
            if not self.storage_path or not os.path.exists(self.storage_path):
                return
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    repos_dict = data.get("repositories", {})
                    for key, val in repos_dict.items():
                        if isinstance(val, dict):
                            meta = RepositoryMetadata.from_dict(val)
                            self._repos[_normalize_repo_id(meta.owner, meta.repo)] = meta
                logger.info(f"Loaded {len(self._repos)} repositories from registry at {self.storage_path}")
            except Exception as e:
                logger.warning(f"Failed to load registry from {self.storage_path}: {e}")

    def _save(self) -> None:
        """Atomically persist registry data to disk."""
        if not self.storage_path:
            return

        with self._lock:
            try:
                dir_name = os.path.dirname(os.path.abspath(self.storage_path))
                os.makedirs(dir_name, exist_ok=True)

                payload = {
                    "version": 1,
                    "repositories": {k: v.to_dict() for k, v in self._repos.items()},
                }

                # Write to temp file in same directory for atomic replace
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=dir_name,
                    delete=False,
                    encoding="utf-8",
                    suffix=".tmp",
                )
                try:
                    json.dump(payload, temp_file, indent=2)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_file.close()
                    os.replace(temp_file.name, self.storage_path)
                except Exception:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                    raise
            except Exception as e:
                logger.error(f"Failed to persist repository registry to {self.storage_path}: {e}")

    def register(self, metadata: RepositoryMetadata) -> None:
        """
        Register or update metadata for a repository.
        """
        with self._lock:
            key = _normalize_repo_id(metadata.owner, metadata.repo)
            self._repos[key] = metadata
            self._save()

    def get(self, owner_or_id: str, repo: str | None = None) -> RepositoryMetadata | None:
        """
        Retrieve repository metadata by 'owner, repo' or 'owner/repo'.
        """
        with self._lock:
            key = _normalize_repo_id(owner_or_id, repo)
            return self._repos.get(key)

    def list_repos(self) -> list[RepositoryMetadata]:
        """
        List all registered repositories.
        """
        with self._lock:
            return list(self._repos.values())

    def delete(self, owner_or_id: str, repo: str | None = None) -> bool:
        """
        Remove a repository from the registry. Returns True if found and deleted.
        """
        with self._lock:
            key = _normalize_repo_id(owner_or_id, repo)
            if key in self._repos:
                del self._repos[key]
                self._save()
                return True
            return False

    def exists(self, owner_or_id: str, repo: str | None = None) -> bool:
        """
        Check if a repository is registered.
        """
        with self._lock:
            key = _normalize_repo_id(owner_or_id, repo)
            return key in self._repos

    def update_status(
        self,
        owner: str,
        repo: str,
        status: str,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """
        Update the operational status of an existing repository.
        """
        with self._lock:
            key = _normalize_repo_id(owner, repo)
            meta = self._repos.get(key)
            if meta:
                meta.status = status
                if error is not None:
                    meta.error = error
                if chunk_count is not None:
                    meta.chunk_count = chunk_count
                self._save()

    def acquire_repo_lock(self, owner: str, repo: str, blocking: bool = True) -> bool:
        """
        Acquire a per-repo lock to prevent concurrent redundant indexing jobs.
        """
        key = _normalize_repo_id(owner, repo)
        with self._repo_locks_guard:
            if key not in self._repo_locks:
                self._repo_locks[key] = threading.Lock()
            repo_lock = self._repo_locks[key]
        return repo_lock.acquire(blocking=blocking)

    def release_repo_lock(self, owner: str, repo: str) -> None:
        """
        Release the per-repo lock.
        """
        key = _normalize_repo_id(owner, repo)
        with self._repo_locks_guard:
            repo_lock = self._repo_locks.get(key)
        if repo_lock and repo_lock.locked():
            try:
                repo_lock.release()
            except RuntimeError:
                pass

    def clear(self) -> None:
        """
        Clear all repository entries from memory and disk.
        """
        with self._lock:
            self._repos.clear()
            self._save()


# Default singleton instance
_default_registry: RepoRegistry | None = None
_default_registry_lock = threading.Lock()


def get_repo_registry(storage_path: str | None = None) -> RepoRegistry:
    """
    Get or create the global repository registry instance.
    """
    global _default_registry
    if storage_path is not None:
        return RepoRegistry(storage_path=storage_path)

    with _default_registry_lock:
        if _default_registry is None:
            _default_registry = RepoRegistry(storage_path=DEFAULT_REGISTRY_PATH)
        return _default_registry
