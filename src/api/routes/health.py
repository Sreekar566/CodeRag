"""
Health, liveness, and readiness endpoints for the intelligence platform.
Provides Kubernetes-compliant and cloud-native container orchestration probes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from src.api.dependencies import get_llm, get_registry, get_settings, get_storage
from src.api.models import (
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
    SubsystemCheck,
)
from src.config import Settings
from src.llm.base import LLMProvider
from src.registry.registry import RepoRegistry
from src.storage.base import VectorStore

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System health and status")
def health_check(
    registry: RepoRegistry = Depends(get_registry),
    storage: VectorStore = Depends(get_storage),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_settings),
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
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        vector_store_type=storage.__class__.__name__,
        llm_provider=llm.__class__.__name__,
        llm_model=str(model_name),
        indexed_repos_count=len(repos),
        collections_count=len(collections),
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness probe for container orchestrators",
)
@router.get(
    "/healthz/live",
    response_model=LivenessResponse,
    include_in_schema=False,
)
def liveness_probe() -> LivenessResponse:
    """
    Lightweight liveness probe for Kubernetes and container orchestrators.
    Returns HTTP 200 as long as the application process is running and responding.
    """
    return LivenessResponse(
        status="alive",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe for container orchestrators",
    responses={
        200: {"description": "Service is ready to handle queries"},
        503: {"description": "Service is unhealthy or unready to accept queries"},
    },
)
@router.get(
    "/healthz/ready",
    response_model=ReadinessResponse,
    include_in_schema=False,
)
def readiness_probe(
    registry: RepoRegistry = Depends(get_registry),
    storage: VectorStore = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> Any:
    """
    Deep readiness probe for Kubernetes and container orchestrators.
    Verifies vector storage connectivity, repository registry accessibility,
    and persistence directory writeability. Returns HTTP 503 if any subsystem fails.
    """
    checks: dict[str, SubsystemCheck] = {}
    is_healthy = True

    # 1. Vector storage connectivity check
    try:
        collections = storage.list_collections()
        checks["storage"] = SubsystemCheck(
            status="healthy",
            details={
                "type": storage.__class__.__name__,
                "collections_count": len(collections),
            },
        )
    except Exception as exc:
        is_healthy = False
        checks["storage"] = SubsystemCheck(
            status="unhealthy",
            error=str(exc),
            details={"type": storage.__class__.__name__},
        )

    # 2. Repository registry accessibility check
    try:
        repos = registry.list_repos()
        checks["registry"] = SubsystemCheck(
            status="healthy",
            details={
                "path": getattr(registry, "storage_path", "unknown"),
                "repos_count": len(repos),
            },
        )
    except Exception as exc:
        is_healthy = False
        checks["registry"] = SubsystemCheck(
            status="unhealthy",
            error=str(exc),
            details={"path": getattr(registry, "storage_path", "unknown")},
        )

    # 3. Persistent filesystem writeability check
    try:
        persist_dir = Path(settings.chroma_persist_directory)
        if persist_dir.exists():
            writable = os.access(persist_dir, os.W_OK)
        else:
            curr = persist_dir
            while not curr.exists() and curr.parent != curr:
                curr = curr.parent
            writable = os.access(curr, os.W_OK)

        if not writable:
            is_healthy = False
            checks["persistence"] = SubsystemCheck(
                status="unhealthy",
                error=f"Persistent directory '{persist_dir}' is not writable",
                details={"path": str(persist_dir)},
            )
        else:
            checks["persistence"] = SubsystemCheck(
                status="healthy",
                details={"path": str(persist_dir), "writable": True},
            )
    except Exception as exc:
        is_healthy = False
        checks["persistence"] = SubsystemCheck(
            status="unhealthy",
            error=str(exc),
        )

    response_data = ReadinessResponse(
        status="ready" if is_healthy else "unhealthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks=checks,
    )

    if not is_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_data.model_dump(),
        )

    return response_data
