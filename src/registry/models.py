"""
Data models for repository metadata, incremental changesets, and indexing results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepositoryMetadata:
    """
    Metadata describing an indexed repository in the intelligence platform.
    """
    owner: str
    repo: str
    collection_name: str
    branch: str = "main"
    commit_sha: str | None = None
    indexed_at: str | None = None
    file_count: int = 0
    chunk_count: int = 0
    file_shas: dict[str, str] = field(default_factory=dict)
    status: str = "ready"  # "ready", "indexing", "failed"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def repo_id(self) -> str:
        """Normalized unique repository identifier 'owner/repo'."""
        return f"{self.owner}/{self.repo}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return {
            "owner": self.owner,
            "repo": self.repo,
            "repo_id": self.repo_id,
            "collection_name": self.collection_name,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "indexed_at": self.indexed_at,
            "file_count": self.file_count,
            "chunk_count": self.chunk_count,
            "file_shas": self.file_shas,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepositoryMetadata:
        """Deserialize metadata from a dictionary."""
        owner = data.get("owner", "")
        repo = data.get("repo", "")
        default_collection = f"{owner}_{repo}" if owner and repo else "codebase"
        return cls(
            owner=owner,
            repo=repo,
            collection_name=data.get("collection_name", default_collection),
            branch=data.get("branch", "main"),
            commit_sha=data.get("commit_sha"),
            indexed_at=data.get("indexed_at"),
            file_count=data.get("file_count", 0),
            chunk_count=data.get("chunk_count", 0),
            file_shas=data.get("file_shas", {}),
            status=data.get("status", "ready"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChangeSet:
    """
    Represents differences detected between repository tree states.
    """
    added_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    unchanged_files: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """True if any files were added, modified, or deleted."""
        return bool(self.added_files or self.modified_files or self.deleted_files)

    @property
    def total_files(self) -> int:
        """Total number of files compared."""
        return (
            len(self.added_files)
            + len(self.modified_files)
            + len(self.deleted_files)
            + len(self.unchanged_files)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize changeset to a dictionary."""
        return {
            "added_files": self.added_files,
            "modified_files": self.modified_files,
            "deleted_files": self.deleted_files,
            "unchanged_files": self.unchanged_files,
            "has_changes": self.has_changes,
            "total_files": self.total_files,
        }


@dataclass
class IncrementalIndexResult:
    """
    Result returned by the incremental indexing pipeline.
    """
    owner: str
    repo: str
    collection: str
    branch: str
    commit_sha: str | None
    status: str  # "created", "updated", "unchanged", "failed"
    files_processed: int
    chunks_stored: int
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_unchanged: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    total_tree_files: int = 0
    filtered_files: int = 0
    skipped_files: list[str] = field(default_factory=list)
    error: str | None = None
    audit_log: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary."""
        return {
            "owner": self.owner,
            "repo": self.repo,
            "collection": self.collection,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "status": self.status,
            "files_processed": self.files_processed,
            "chunks_stored": self.chunks_stored,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "files_unchanged": self.files_unchanged,
            "chunks_added": self.chunks_added,
            "chunks_deleted": self.chunks_deleted,
            "total_tree_files": self.total_tree_files,
            "filtered_files": self.filtered_files,
            "skipped_files": self.skipped_files,
            "error": self.error,
            "audit_log": self.audit_log,
        }
