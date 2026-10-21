"""
FastAPI application factory and middleware configuration.
"""
from __future__ import annotations

import logging
from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import (
    health_router,
    repos_router,
    index_router,
    query_router,
    collections_router,
    observability_router,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application instance.
    """
    app = FastAPI(
        title="AI Codebase Intelligence API",
        description=(
            "REST API for repository ingestion, incremental re-indexing, "
            "semantic retrieval, and citation-grounded RAG codebase Q&A."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Enable CORS for frontend clients (Streamlit, React, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
                "repos": "/repos",
                "index": "/index",
                "query": "/query",
                "collections": "/collections",
                "metrics": "/metrics",
                "query_logs": "/metrics/queries",
                "audit": "/repos/{owner}/{repo}/audit",
            },
        }

    return app


# Default application instance for ASGI servers (e.g. uvicorn)
app = create_app()
