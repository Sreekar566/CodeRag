"""
Health and status endpoint for the intelligence platform.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from src.api.dependencies import get_registry, get_storage, get_llm
from src.api.models import HealthResponse
from src.registry.registry import RepoRegistry
from src.storage.base import VectorStore
from src.llm.base import LLMProvider

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System health and status")
def health_check(
    registry: RepoRegistry = Depends(get_registry),
    storage: VectorStore = Depends(get_storage),
    llm: LLMProvider = Depends(get_llm),
) -> HealthResponse:
    """
    Returns system status, configured vector store, active LLM provider,
    and repository/collection counts.
    """
    model_name = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
    collections = storage.list_collections()
    repos = registry.list_repos()

    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        vector_store_type=storage.__class__.__name__,
        llm_provider=llm.__class__.__name__,
        llm_model=str(model_name),
        indexed_repos_count=len(repos),
        collections_count=len(collections),
    )
