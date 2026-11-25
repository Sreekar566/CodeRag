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

    def _resolve_target_collection(self, owner: str, repo: str, collection_name: str | None) -> str:
        """Resolve and sanitize target collection name."""
        if collection_name is None:
            collection_name = f"{owner}_{repo}"
        return sanitize_collection_name(collection_name)

    def _resolve_branch(self, owner: str, repo: str, token: str | None, branch: str | None) -> str:
        """Resolve branch, auto-detecting default remote branch when required."""
        resolved = branch
        if not resolved or resolved == "auto":
            try:
                resolved = get_default_branch(owner, repo, token)
            except Exception as e:
                logger.warning(
                    f"Could not auto-detect default branch for {owner}/{repo}: {e}. Falling back to 'main'"
                )
                resolved = "main"
        return resolved

    def _get_latest_commit_sha(self, owner: str, repo: str, token: str | None, branch: str) -> str | None:
        """Attempt to fetch the latest commit SHA for the target branch."""
        try:
            return get_latest_commit_sha(owner, repo, token, branch=branch)
        except Exception as e:
            logger.warning(f"Could not retrieve latest commit SHA for {owner}/{repo}@{branch}: {e}")
            return None

    def _build_index_result(
        self,
        owner: str,
        repo: str,
        target_collection: str,
        branch: str,
        latest_commit_sha: str | None,
        status: str,
        files_processed: int,
        chunks_stored: int,
        files_added: int,
        files_modified: int,
        files_deleted: int,
        files_unchanged: int,
        total_tree_files: int,
        filtered_files: int,
        skipped_files: list[str],
        audit_log: Any = None,
    ) -> IncrementalIndexResult:
        """Construct a standardized IncrementalIndexResult."""
        audit_dict = (
            audit_log.to_dict()
            if hasattr(audit_log, "to_dict")
            else (audit_log if isinstance(audit_log, dict) else None)
        )
        return IncrementalIndexResult(
            owner=owner,
            repo=repo,
            collection=target_collection,
            branch=branch,
            commit_sha=latest_commit_sha,
            status=status,
            files_processed=files_processed,
            chunks_stored=chunks_stored,
            files_added=files_added,
            files_modified=files_modified,
            files_deleted=files_deleted,
            files_unchanged=files_unchanged,
            chunks_added=chunks_stored,
            chunks_deleted=files_deleted + files_modified,
            total_tree_files=total_tree_files,
            filtered_files=filtered_files,
            skipped_files=skipped_files,
            audit_log=audit_dict,
        )

    def _check_unchanged_fast_path(
        self,
        owner: str,
        repo: str,
        target_collection: str,
        branch: str,
        latest_commit_sha: str | None,
        prev_meta: RepositoryMetadata | None,
        force: bool,
    ) -> IncrementalIndexResult | None:
        """
        Fast-path check: if repository is already ready and latest remote commit SHA matches,
        skip tree fetching and processing entirely.
        """
        if (
            not force
            and prev_meta is not None
            and prev_meta.status == "ready"
            and latest_commit_sha is not None
            and prev_meta.commit_sha == latest_commit_sha
        ):
            logger.info(
                f"Repository {owner}/{repo}@{branch} is unchanged at commit {latest_commit_sha}."
            )
            return self._build_index_result(
                owner=owner,
                repo=repo,
                target_collection=target_collection,
                branch=branch,
                latest_commit_sha=latest_commit_sha,
                status="unchanged",
                files_processed=0,
                chunks_stored=0,
                files_added=0,
                files_modified=0,
                files_deleted=0,
                files_unchanged=prev_meta.file_count,
                total_tree_files=prev_meta.file_count,
                filtered_files=prev_meta.file_count,
                skipped_files=[],
            )
        return None

    def _fetch_and_audit_tree(
        self,
        owner: str,
        repo: str,
        token: str | None,
        branch: str,
        latest_commit_sha: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, str]]:
        """
        Fetch repository tree from GitHub and filter out noise files with audit logging.
        Returns (tree, filtered_items, audit_log, current_file_shas).
        """
        tree = get_repo_tree(owner, repo, token, branch=branch)
        repo_id = f"{owner}/{repo}"
        filtered_items, audit_log = audit_filter_files(
            tree,
            repo_id=repo_id,
            branch=branch,
            commit_sha=latest_commit_sha,
        )
        get_telemetry_collector().record_audit(audit_log)

        current_file_shas = {
            item["path"]: item.get("sha", "")
            for item in filtered_items
            if "path" in item
        }
        return tree, filtered_items, audit_log, current_file_shas

    def _purge_chunks_for_files(self, target_collection: str, paths: list[str]) -> None:
        """Delete stored vector chunks for modified or deleted files."""
        for path in paths:
            try:
                self.vector_store.delete(
                    where={"path": path},
                    collection_name=target_collection,
                )
            except Exception as e:
                logger.warning(f"Failed to delete old chunks for {path}: {e}")

    def _determine_changes_and_cleanup(
        self,
        target_collection: str,
        filtered_items: list[dict[str, Any]],
        prev_meta: RepositoryMetadata | None,
        force: bool,
        latest_commit_sha: str | None = None,
        purge_modified: bool = True,
    ) -> tuple[list[str], list[str], list[str], list[str], int, bool, bool]:
        """
        Compute file diffs against previously indexed SHAs and purge obsolete chunks.
        Returns:
            (files_to_fetch, files_added, files_modified, files_deleted, unchanged_count, is_initial_index, is_delta_unchanged)
        """
        is_initial_index = force or (prev_meta is None) or (not prev_meta.file_shas)

        if is_initial_index:
            if force:
                try:
                    self.vector_store.clear(target_collection)
                except Exception as e:
                    logger.warning(f"Failed to clear collection {target_collection}: {e}")
            files_to_fetch = [item["path"] for item in filtered_items if "path" in item]
            return files_to_fetch, list(files_to_fetch), [], [], 0, True, False

        changes = detect_changes(filtered_items, prev_meta.file_shas)
        if not changes.has_changes and prev_meta.status == "ready" and latest_commit_sha == prev_meta.commit_sha:
            return [], [], [], [], len(changes.unchanged_files), False, True

        files_deleted = changes.deleted_files
        files_modified = changes.modified_files
        files_added = changes.added_files
        unchanged_count = len(changes.unchanged_files)
        files_to_fetch = files_added + files_modified

        # Remove chunks for deleted (and optionally modified) files from vector store
        to_purge = files_deleted + files_modified if purge_modified else files_deleted
        if to_purge:
            self._purge_chunks_for_files(target_collection, to_purge)

        return (
            files_to_fetch,
            files_added,
            files_modified,
            files_deleted,
            unchanged_count,
            False,
            False,
        )

    def _fetch_file_contents_with_failures(
        self,
        owner: str,
        repo: str,
        token: str | None,
        files_to_fetch: list[str],
    ) -> tuple[list[dict[str, str]], list[str], list[str]]:
        """
        Fetch file contents from GitHub REST API for files to process.
        Returns (fetched_files, skipped_files, failed_files) where failed_files
        tracks paths that raised an error during fetch (as opposed to binary files).
        """
        fetched_files: list[dict[str, str]] = []
        skipped_files: list[str] = []
        failed_files: list[str] = []

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
                failed_files.append(path)

        return fetched_files, skipped_files, failed_files

    def _fetch_file_contents(
        self,
        owner: str,
        repo: str,
        token: str | None,
        files_to_fetch: list[str],
    ) -> tuple[list[dict[str, str]], list[str]]:
        """
        Fetch file contents from GitHub REST API for files to process.
        Returns (fetched_files, skipped_files).
        """
        fetched_files, skipped_files, _ = self._fetch_file_contents_with_failures(
            owner=owner,
            repo=repo,
            token=token,
            files_to_fetch=files_to_fetch,
        )
        return fetched_files, skipped_files

    def _chunk_embed_and_store(
        self,
        fetched_files: list[dict[str, str]],
        target_collection: str,
        model: Any,
        chunk_fn: Callable[[str, str], list[dict]],
    ) -> int:
        """
        Chunk new and updated files, generate embeddings, and store them in the vector database.
        Returns the number of chunks stored.
        """
        chunks_to_add: list[dict] = []
        for f in fetched_files:
            chunks = chunk_fn(f["content"], f["path"])
            chunks_to_add.extend(chunks)

        if not chunks_to_add:
            return 0

        embedded = embed_chunks(chunks_to_add, model=model)
        return self.vector_store.add_chunks(
            embedded,
            collection_name=target_collection,
        )

    def _update_registry_metadata(
        self,
        owner: str,
        repo: str,
        target_collection: str,
        branch: str,
        latest_commit_sha: str | None,
        file_count: int,
        chunk_count: int,
        current_file_shas: dict[str, str],
        audit_log: Any,
    ) -> None:
        """Update repository metadata in the persistent registry."""
        now_iso = datetime.now(timezone.utc).isoformat()
        audit_dict = (
            audit_log.to_dict()
            if hasattr(audit_log, "to_dict")
            else (audit_log if isinstance(audit_log, dict) else None)
        )
        new_meta = RepositoryMetadata(
            owner=owner,
            repo=repo,
            collection_name=target_collection,
            branch=branch,
            commit_sha=latest_commit_sha,
            indexed_at=now_iso,
            file_count=file_count,
            chunk_count=chunk_count,
            file_shas=current_file_shas,
            status="ready",
            error=None,
            metadata={"audit_log": audit_dict} if audit_dict is not None else {},
        )
        self.registry.register(new_meta)

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
        target_collection = self._resolve_target_collection(owner, repo, collection_name)
        resolved_branch = self._resolve_branch(owner, repo, token, branch)

        # Concurrency control: acquire per-repository lock
        locked = self.registry.acquire_repo_lock(owner, repo, blocking=True)
        if not locked:
            raise RuntimeError(f"Could not acquire lock for repository {owner}/{repo}; indexing already in progress.")

        try:
            # Step 1 & 2: Remote commit detection and fast-path check
            latest_commit_sha = self._get_latest_commit_sha(owner, repo, token, resolved_branch)
            prev_meta = self.registry.get(owner, repo)

            unchanged_result = self._check_unchanged_fast_path(
                owner=owner,
                repo=repo,
                target_collection=target_collection,
                branch=resolved_branch,
                latest_commit_sha=latest_commit_sha,
                prev_meta=prev_meta,
                force=force,
            )
            if unchanged_result is not None:
                return unchanged_result

            # Mark repository as indexing
            self.registry.update_status(owner, repo, status="indexing")

            # Step 3: Fetch file tree and filter with transparency audit
            tree, filtered_items, audit_log, current_file_shas = self._fetch_and_audit_tree(
                owner=owner,
                repo=repo,
                token=token,
                branch=resolved_branch,
                latest_commit_sha=latest_commit_sha,
            )

            # Step 4: Determine file changes and purge deleted files
            (
                files_to_fetch,
                files_added,
                files_modified,
                files_deleted,
                unchanged_count,
                is_initial_index,
                is_delta_unchanged,
            ) = self._determine_changes_and_cleanup(
                target_collection=target_collection,
                filtered_items=filtered_items,
                prev_meta=prev_meta,
                force=force,
                latest_commit_sha=latest_commit_sha,
                purge_modified=False,
            )

            if is_delta_unchanged:
                self.registry.update_status(owner, repo, status="ready")
                return self._build_index_result(
                    owner=owner,
                    repo=repo,
                    target_collection=target_collection,
                    branch=resolved_branch,
                    latest_commit_sha=latest_commit_sha,
                    status="unchanged",
                    files_processed=0,
                    chunks_stored=0,
                    files_added=0,
                    files_modified=0,
                    files_deleted=0,
                    files_unchanged=unchanged_count,
                    total_tree_files=len(tree),
                    filtered_files=len(filtered_items),
                    skipped_files=[],
                    audit_log=audit_log,
                )

            # Step 5: Fetch content for delta files
            fetched_files, skipped_files, failed_files = self._fetch_file_contents_with_failures(
                owner=owner,
                repo=repo,
                token=token,
                files_to_fetch=files_to_fetch,
            )
            failed_set = set(failed_files)

            # Purge old chunks for modified files only when they did not fail to fetch
            # (so transient fetch errors do not destroy previously indexed chunks)
            modified_to_purge = [p for p in files_modified if p not in failed_set]
            if modified_to_purge:
                self._purge_chunks_for_files(target_collection, modified_to_purge)

            # Step 6: Chunk, embed, and store delta chunks
            model = embedder_model if embedder_model is not None else self.embedder_model
            chunks_stored = self._chunk_embed_and_store(
                fetched_files=fetched_files,
                target_collection=target_collection,
                model=model,
                chunk_fn=chunk_fn,
            )
            total_chunks = self.vector_store.count(target_collection)

            # Step 7: Update registry record without poisoning SHAs for failed fetches
            if failed_set:
                effective_prev_shas = {} if is_initial_index else (prev_meta.file_shas if prev_meta else {})
                recorded_file_shas: dict[str, str] = {}
                for path, sha in current_file_shas.items():
                    if path in failed_set:
                        if path in effective_prev_shas:
                            recorded_file_shas[path] = effective_prev_shas[path]
                    else:
                        recorded_file_shas[path] = sha
                recorded_commit_sha = None if is_initial_index else (prev_meta.commit_sha if prev_meta else None)
            else:
                recorded_file_shas = current_file_shas
                recorded_commit_sha = latest_commit_sha

            self._update_registry_metadata(
                owner=owner,
                repo=repo,
                target_collection=target_collection,
                branch=resolved_branch,
                latest_commit_sha=recorded_commit_sha,
                file_count=len(recorded_file_shas),
                chunk_count=total_chunks,
                current_file_shas=recorded_file_shas,
                audit_log=audit_log,
            )

            # Step 8: Assemble final index result
            status_str = "created" if is_initial_index else "updated"
            return self._build_index_result(
                owner=owner,
                repo=repo,
                target_collection=target_collection,
                branch=resolved_branch,
                latest_commit_sha=recorded_commit_sha,
                status=status_str,
                files_processed=len(fetched_files),
                chunks_stored=chunks_stored,
                files_added=len(files_added),
                files_modified=len(files_modified),
                files_deleted=len(files_deleted),
                files_unchanged=unchanged_count,
                total_tree_files=len(tree),
                filtered_files=len(filtered_items),
                skipped_files=skipped_files,
                audit_log=audit_log,
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
