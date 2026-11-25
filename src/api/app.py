"""
FastAPI application factory and middleware configuration.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RequestTracingMiddleware
from src.api.routes import (
    health_router,
    repos_router,
    index_router,
    query_router,
    collections_router,
    observability_router,
)
from src.config import get_settings
from src.observability.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application startup and shutdown lifecycle events.
    Re-applies unified logging (including uvicorn loggers) and emits safe startup metadata.
    """
    settings = get_settings()
    setup_logging(force=True)
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"(llm_provider={settings.effective_llm_provider}, log_format={settings.log_format})",
        extra={
            "event": "startup",
            "version": settings.app_version,
            "llm_provider": settings.effective_llm_provider,
            "embedding_model": settings.embedding_model,
            "log_level": settings.log_level,
            "log_format": settings.log_format,
            "chroma_persist_directory": settings.chroma_persist_directory,
        },
    )
    try:
        yield
    finally:
        logger.info(
            f"Shutting down {settings.app_name} v{settings.app_version}",
            extra={"event": "shutdown", "version": settings.app_version},
        )


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application instance.
    """
    settings = get_settings()

    # Configure unified application logging
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "REST API for repository ingestion, incremental re-indexing, "
            "semantic retrieval, and citation-grounded RAG codebase Q&A."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    # Request correlation, access logging, and process timing middleware (inner)
    app.add_middleware(RequestTracingMiddleware)

    # Enable CORS for frontend clients (Streamlit, React, etc.) (outer)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # Register API routers
    app.include_router(health_router)
    app.include_router(repos_router)
    app.include_router(index_router)
    app.include_router(query_router)
    app.include_router(collections_router)
    app.include_router(observability_router)

    # Root welcome route
    @app.get("/", tags=["General"], summary="API Root Overview")
    def root() -> dict[str, Any]:
        return {
            "name": "AI Codebase Intelligence API",
            "version": "0.1.0",
            "status": "online",
            "documentation": "/docs",
            "endpoints": {
                "health": "/health",
                "liveness": "/health/live",
                "readiness": "/health/ready",
                "repos": "/repos",
                "index": "/index",
                "query": "/query",
                "collections": "/collections",
                "metrics": "/metrics",
                "prometheus": "/metrics/prometheus",
                "query_logs": "/metrics/queries",
                "audit": "/repos/{owner}/{repo}/audit",
            },
        }

    return app


# Default application instance for ASGI servers (e.g. uvicorn)
app = create_app()
