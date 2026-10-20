"""
Repository indexing endpoint supporting synchronous and background execution.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status

from src.api.dependencies import get_indexer, get_registry
from src.api.models import IndexRequest, IndexResponse
from src.registry.incremental import IncrementalIndexer
from src.registry.registry import RepoRegistry
from src.storage.base import sanitize_collection_name

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Indexing"])


def _execute_background_indexing(
    indexer: IncrementalIndexer,
    req: IndexRequest,
) -> None:
    """Worker function executed by FastAPI BackgroundTasks."""
    logger.info(f"Starting background indexing for {req.owner}/{req.repo}@{req.branch}")
    try:
        indexer.index_repository(
            owner=req.owner,
            repo=req.repo,
            token=req.token,
            branch=req.branch,
            collection_name=req.collection_name,
            force=req.force,
        )
        logger.info(f"Background indexing completed for {req.owner}/{req.repo}")
    except Exception as e:
        logger.error(f"Background indexing failed for {req.owner}/{req.repo}: {e}", exc_info=True)


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Index or incrementally re-index a GitHub repository",
)
def index_repository(
    request: IndexRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    indexer: IncrementalIndexer = Depends(get_indexer),
    registry: RepoRegistry = Depends(get_registry),
) -> IndexResponse:
    """
    Ingest, filter, chunk, embed, and index a GitHub repository into the vector store.
    
    If the repository has already been indexed at the current commit SHA, unchanged files
    are skipped. If changes are detected, only modified/added files are processed and deleted
    files are purged.
    
    If `background=true` is requested, indexing is dispatched as a background task and returns
    HTTP 202 Accepted immediately.
    """
    owner = request.owner.strip()
    repo = request.repo.strip()
    target_collection = sanitize_collection_name(
        request.collection_name or f"{owner}_{repo}"
    )

    # Check if indexing is already active for this repo
    can_lock = registry.acquire_repo_lock(owner, repo, blocking=False)
    if not can_lock:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{owner}/{repo}' is currently being indexed by another process.",
        )
    registry.release_repo_lock(owner, repo)

    # Handle background indexing
    if request.background:
        # Mark repo as indexing in registry
        registry.update_status(owner, repo, status="indexing")
        background_tasks.add_task(_execute_background_indexing, indexer, request)
        response.status_code = status.HTTP_202_ACCEPTED
        return IndexResponse(
            owner=owner,
            repo=repo,
            collection=target_collection,
            branch=request.branch or "main",
            commit_sha=None,
            status="indexing",
            files_processed=0,
            chunks_stored=0,
            background_task=True,
            message=f"Indexing job for '{owner}/{repo}' initiated in the background.",
        )

    # Synchronous indexing
    try:
        result = indexer.index_repository(
            owner=owner,
            repo=repo,
            token=request.token,
            branch=request.branch,
            collection_name=request.collection_name,
            force=request.force,
        )
    except RuntimeError as e:
        if "indexing already in progress" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error indexing repository {owner}/{repo}: {err_msg}", exc_info=True)
        if "not found" in err_msg.lower() or "404" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository '{owner}/{repo}' or branch '{request.branch}' not found on GitHub: {err_msg}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index repository '{owner}/{repo}': {err_msg}",
        )

    return IndexResponse(
        owner=result.owner,
        repo=result.repo,
        collection=result.collection,
        branch=result.branch,
        commit_sha=result.commit_sha,
        status=result.status,
        files_processed=result.files_processed,
        chunks_stored=result.chunks_stored,
        files_added=result.files_added,
        files_modified=result.files_modified,
        files_deleted=result.files_deleted,
        files_unchanged=result.files_unchanged,
        chunks_added=result.chunks_added,
        chunks_deleted=result.chunks_deleted,
        total_tree_files=result.total_tree_files,
        filtered_files=result.filtered_files,
        skipped_files=result.skipped_files,
        error=result.error,
        background_task=False,
    )
