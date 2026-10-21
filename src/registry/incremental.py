"""
Incremental indexing pipeline with Git tree SHA change detection and vector synchronization.
Only fetches, chunks, and embeds files that are added or modified, and purges deleted files.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Callable

from src.ingestion.github_client import (
    get_repo_tree,
    get_file_content,
    get_default_branch,
    get_latest_commit_sha,
)
from src.ingestion.filters import filter_files, audit_filter_files
from src.ingestion.chunker import chunk_text
from src.embedding.embedder import embed_chunks
from src.storage.base import VectorStore, sanitize_collection_name
from src.storage.chroma_store import get_vector_store
from src.registry.models import ChangeSet, IncrementalIndexResult, RepositoryMetadata
from src.registry.registry import RepoRegistry, get_repo_registry
from src.observability.collector import get_telemetry_collector

logger = logging.getLogger(__name__)


def detect_changes(
    current_filtered_tree: list[dict[str, Any]],
    previous_file_shas: dict[str, str],
) -> ChangeSet:
    """
    Compare current repository file tree items with previously recorded file SHAs.
    Returns a ChangeSet categorizing files into added, modified, deleted, and unchanged.
    """
    current_paths: dict[str, str] = {
        item["path"]: item.get("sha", "")
        for item in current_filtered_tree
        if "path" in item
    }

    added_files: list[str] = sorted(
        [p for p in current_paths if p not in previous_file_shas]
    )
    deleted_files: list[str] = sorted(
        [p for p in previous_file_shas if p not in current_paths]
    )
    modified_files: list[str] = sorted(
        [
            p
            for p in current_paths
            if p in previous_file_shas and current_paths[p] != previous_file_shas[p]
        ]
    )
    unchanged_files: list[str] = sorted(
        [
            p
            for p in current_paths
            if p in previous_file_shas and current_paths[p] == previous_file_shas[p]
        ]
    )

    return ChangeSet(
        added_files=added_files,
        modified_files=modified_files,
        deleted_files=deleted_files,
        unchanged_files=unchanged_files,
    )


class IncrementalIndexer:
    """
    Coordinates incremental repository indexing and synchronization with VectorStore.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        registry: RepoRegistry | None = None,
        embedder_model: Any = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.registry = registry or get_repo_registry()
        self.embedder_model = embedder_model

    def index_repository(
        self,
        owner: str,
        repo: str,
        token: str | None = None,
        branch: str | None = "main",
        collection_name: str | None = None,
        force: bool = False,
        embedder_model: Any = None,
        chunk_fn: Callable[[str, str], list[dict]] = chunk_text,
    ) -> IncrementalIndexResult:
        """
        Incrementally index a repository.
        If the repo is already indexed and unchanged at the latest commit, re-indexing is skipped.
        If files were modified or deleted, only the delta is processed and synchronized.
        """
        # Resolve target collection name
        if collection_name is None:
            collection_name = f"{owner}_{repo}"
        target_collection = sanitize_collection_name(collection_name)

        # Resolve branch
        resolved_branch = branch
        if not resolved_branch or resolved_branch == "auto":
            try:
                resolved_branch = get_default_branch(owner, repo, token)
            except Exception as e:
                logger.warning(f"Could not auto-detect default branch for {owner}/{repo}: {e}. Falling back to 'main'")
                resolved_branch = "main"

        # Concurrency control: acquire per-repository lock
        locked = self.registry.acquire_repo_lock(owner, repo, blocking=True)
        if not locked:
            raise RuntimeError(f"Could not acquire lock for repository {owner}/{repo}; indexing already in progress.")

        try:
            # Step 1: Detect latest commit SHA
            latest_commit_sha: str | None = None
            try:
                latest_commit_sha = get_latest_commit_sha(owner, repo, token, branch=resolved_branch)
            except Exception as e:
                logger.warning(f"Could not retrieve latest commit SHA for {owner}/{repo}@{resolved_branch}: {e}")

            # Step 2: Check previous indexed state in registry
            prev_meta = self.registry.get(owner, repo)

            # Fast-path: commit SHA matches and not forcing full reindex
            if (
                not force
                and prev_meta is not None
                and prev_meta.status == "ready"
                and latest_commit_sha is not None
                and prev_meta.commit_sha == latest_commit_sha
            ):
                logger.info(
                    f"Repository {owner}/{repo}@{resolved_branch} is unchanged at commit {latest_commit_sha}."
                )
                return IncrementalIndexResult(
                    owner=owner,
                    repo=repo,
                    collection=target_collection,
                    branch=resolved_branch,
                    commit_sha=latest_commit_sha,
                    status="unchanged",
                    files_processed=0,
                    chunks_stored=0,
                    files_added=0,
                    files_modified=0,
                    files_deleted=0,
                    files_unchanged=prev_meta.file_count,
                    chunks_added=0,
                    chunks_deleted=0,
                    total_tree_files=prev_meta.file_count,
                    filtered_files=prev_meta.file_count,
                    skipped_files=[],
                )

            # Mark repository as indexing
            self.registry.update_status(owner, repo, status="indexing")

            # Step 3: Fetch file tree and filter with transparency audit
            tree = get_repo_tree(owner, repo, token, branch=resolved_branch)
            repo_id = f"{owner}/{repo}"
            filtered_items, audit_log = audit_filter_files(
                tree,
                repo_id=repo_id,
                branch=resolved_branch,
                commit_sha=latest_commit_sha,
            )
            get_telemetry_collector().record_audit(audit_log)

            current_file_shas = {
                item["path"]: item.get("sha", "")
                for item in filtered_items
                if "path" in item
            }

            model = embedder_model if embedder_model is not None else self.embedder_model

            # Step 4: Determine changes
            is_initial_index = force or (prev_meta is None) or (not prev_meta.file_shas)

            files_to_fetch: list[str] = []
            files_deleted: list[str] = []
            files_modified: list[str] = []
            files_added: list[str] = []
            unchanged_count = 0

            if is_initial_index:
                if force:
                    try:
                        self.vector_store.clear(target_collection)
                    except Exception as e:
                        logger.warning(f"Failed to clear collection {target_collection}: {e}")
                files_to_fetch = [item["path"] for item in filtered_items if "path" in item]
                files_added = list(files_to_fetch)
            else:
                changes = detect_changes(filtered_items, prev_meta.file_shas)
                if not changes.has_changes and latest_commit_sha == prev_meta.commit_sha:
                    self.registry.update_status(owner, repo, status="ready")
                    return IncrementalIndexResult(
                        owner=owner,
                        repo=repo,
                        collection=target_collection,
                        branch=resolved_branch,
                        commit_sha=latest_commit_sha,
                        status="unchanged",
                        files_processed=0,
                        chunks_stored=0,
                        files_added=0,
                        files_modified=0,
                        files_deleted=0,
                        files_unchanged=len(changes.unchanged_files),
                        chunks_added=0,
                        chunks_deleted=0,
                        total_tree_files=len(tree),
                        filtered_files=len(filtered_items),
                        skipped_files=[],
                    )

                files_deleted = changes.deleted_files
                files_modified = changes.modified_files
                files_added = changes.added_files
                unchanged_count = len(changes.unchanged_files)
                files_to_fetch = files_added + files_modified

                # Remove chunks for deleted and modified files from vector store
                paths_to_delete = files_deleted + files_modified
                for path in paths_to_delete:
                    try:
                        self.vector_store.delete(
                            where={"path": path},
                            collection_name=target_collection,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to delete old chunks for {path}: {e}")

            # Step 5: Fetch content for added and modified files
            fetched_files: list[dict[str, str]] = []
            skipped_files: list[str] = []

            for path in files_to_fetch:
                try:
                    content = get_file_content(owner, repo, path, token)
                    if content is not None:
                        fetched_files.append({"path": path, "content": content})
                    else:
                        skipped_files.append(path)
                except Exception as e:
                    logger.warning(f"Failed to fetch content for {path}: {e}")
                    skipped_files.append(path)

            # Step 6: Chunk and embed new/updated files
            chunks_to_add: list[dict] = []
            for f in fetched_files:
                path = f["path"]
                content = f["content"]
                chunks = chunk_fn(content, path)
                chunks_to_add.extend(chunks)

            chunks_stored = 0
            if chunks_to_add:
                embedded = embed_chunks(chunks_to_add, model=model)
                chunks_stored = self.vector_store.add_chunks(
                    embedded,
                    collection_name=target_collection,
                )

            total_chunks = self.vector_store.count(target_collection)

            # Step 7: Update registry record
            now_iso = datetime.now(timezone.utc).isoformat()
            new_meta = RepositoryMetadata(
                owner=owner,
                repo=repo,
                collection_name=target_collection,
                branch=resolved_branch,
                commit_sha=latest_commit_sha,
                indexed_at=now_iso,
                file_count=len(filtered_items),
                chunk_count=total_chunks,
                file_shas=current_file_shas,
                status="ready",
                error=None,
                metadata={"audit_log": audit_log.to_dict()},
            )
            self.registry.register(new_meta)

            status_str = "created" if is_initial_index else "updated"

            return IncrementalIndexResult(
                owner=owner,
                repo=repo,
                collection=target_collection,
                branch=resolved_branch,
                commit_sha=latest_commit_sha,
                status=status_str,
                files_processed=len(fetched_files),
                chunks_stored=chunks_stored,
                files_added=len(files_added),
                files_modified=len(files_modified),
                files_deleted=len(files_deleted),
                files_unchanged=unchanged_count,
                chunks_added=chunks_stored,
                chunks_deleted=len(files_deleted) + len(files_modified),
                total_tree_files=len(tree),
                filtered_files=len(filtered_items),
                skipped_files=skipped_files,
                audit_log=audit_log.to_dict(),
            )

        except Exception as e:
            self.registry.update_status(owner, repo, status="failed", error=str(e))
            raise
        finally:
            self.registry.release_repo_lock(owner, repo)


def index_repository_incremental(
    owner: str,
    repo: str,
    token: str | None = None,
    branch: str | None = "main",
    collection_name: str | None = None,
    force: bool = False,
    vector_store: VectorStore | None = None,
    registry: RepoRegistry | None = None,
    embedder_model: Any = None,
) -> IncrementalIndexResult:
    """
    Convenience function to incrementally index a repository.
    """
    indexer = IncrementalIndexer(
        vector_store=vector_store,
        registry=registry,
        embedder_model=embedder_model,
    )
    return indexer.index_repository(
        owner=owner,
        repo=repo,
        token=token,
        branch=branch,
        collection_name=collection_name,
        force=force,
        embedder_model=embedder_model,
    )
