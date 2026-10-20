"""
Endpoints for managing and inspecting indexed repositories.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_registry, get_storage
from src.api.models import (
    ChangeSetResponse,
    DeleteResponse,
    RepoListResponse,
    RepoResponse,
)
from src.registry.registry import RepoRegistry
from src.registry.models import RepositoryMetadata
from src.registry.incremental import detect_changes
from src.storage.base import VectorStore
from src.ingestion.github_client import (
    get_repo_tree,
    get_latest_commit_sha,
    get_default_branch,
)
from src.ingestion.filters import filter_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repos", tags=["Repositories"])


def _to_repo_response(meta: RepositoryMetadata) -> RepoResponse:
    """Helper to convert RepositoryMetadata domain object to Pydantic response."""
    return RepoResponse(
        owner=meta.owner,
        repo=meta.repo,
        repo_id=meta.repo_id,
        collection_name=meta.collection_name,
        branch=meta.branch,
        commit_sha=meta.commit_sha,
        indexed_at=meta.indexed_at,
        file_count=meta.file_count,
        chunk_count=meta.chunk_count,
        status=meta.status,
        error=meta.error,
        metadata=meta.metadata,
    )


@router.get("", response_model=RepoListResponse, summary="List all indexed repositories")
def list_repositories(
    registry: RepoRegistry = Depends(get_registry),
) -> RepoListResponse:
    """
    Retrieve all registered repositories in the platform with their indexing status.
    """
    repos = registry.list_repos()
    repo_responses = [_to_repo_response(r) for r in repos]
    return RepoListResponse(repos=repo_responses, total=len(repo_responses))


@router.get(
    "/{owner}/{repo}",
    response_model=RepoResponse,
    summary="Get repository metadata",
)
def get_repository(
    owner: str,
    repo: str,
    registry: RepoRegistry = Depends(get_registry),
) -> RepoResponse:
    """
    Get detailed metadata for a registered repository by owner and name.
    """
    meta = registry.get(owner, repo)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{owner}/{repo}' is not registered.",
        )
    return _to_repo_response(meta)


@router.delete(
    "/{owner}/{repo}",
    response_model=DeleteResponse,
    summary="Delete repository and its vector collection",
)
def delete_repository(
    owner: str,
    repo: str,
    registry: RepoRegistry = Depends(get_registry),
    storage: VectorStore = Depends(get_storage),
) -> DeleteResponse:
    """
    Remove a repository from the registry and purge its vector collection from storage.
    """
    meta = registry.get(owner, repo)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{owner}/{repo}' not found in registry.",
        )

    # Purge vector collection
    if meta.collection_name:
        try:
            storage.clear(meta.collection_name)
        except Exception as e:
            logger.warning(
                f"Error clearing vector collection {meta.collection_name} during deletion: {e}"
            )

    # Delete from registry
    deleted = registry.delete(owner, repo)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete repository '{owner}/{repo}' from registry.",
        )

    return DeleteResponse(
        success=True,
        message=f"Repository '{owner}/{repo}' and collection '{meta.collection_name}' deleted.",
        target=f"{owner}/{repo}",
    )


@router.get(
    "/{owner}/{repo}/changes",
    response_model=ChangeSetResponse,
    summary="Preview pending Git changes without indexing",
)
def preview_repo_changes(
    owner: str,
    repo: str,
    branch: str | None = Query(default=None, description="Branch to check (defaults to registered branch)"),
    token: str | None = Query(default=None, description="GitHub token override"),
    registry: RepoRegistry = Depends(get_registry),
) -> ChangeSetResponse:
    """
    Inspect pending changes in the remote GitHub repository compared to the local indexed state.
    Computes added, modified, deleted, and unchanged files using Git Merkle tree SHAs.
    """
    meta = registry.get(owner, repo)
    target_branch = branch or (meta.branch if meta else "main")

    try:
        latest_sha = get_latest_commit_sha(owner, repo, token, branch=target_branch)
    except Exception as e:
        latest_sha = None
        logger.warning(f"Could not retrieve latest commit SHA for {owner}/{repo}: {e}")

    try:
        raw_tree = get_repo_tree(owner, repo, token, branch=target_branch)
        filtered_tree = filter_files(raw_tree)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch tree from GitHub for '{owner}/{repo}': {e}",
        )

    prev_shas = meta.file_shas if meta else {}
    changeset = detect_changes(filtered_tree, prev_shas)

    return ChangeSetResponse(
        owner=owner,
        repo=repo,
        branch=target_branch,
        current_commit_sha=meta.commit_sha if meta else None,
        latest_commit_sha=latest_sha,
        added_files=changeset.added_files,
        modified_files=changeset.modified_files,
        deleted_files=changeset.deleted_files,
        unchanged_files=changeset.unchanged_files,
        has_changes=changeset.has_changes,
        total_files=changeset.total_files,
    )
