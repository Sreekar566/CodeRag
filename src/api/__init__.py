"""
Backend API Layer for AI Codebase Intelligence Platform.
"""
from src.api.app import app, create_app
from src.api.middleware import RequestTracingMiddleware
from src.api.models import (
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    RepoResponse,
    RepoListResponse,
    CollectionResponse,
    CollectionListResponse,
)

__all__ = [
    "app",
    "create_app",
    "RequestTracingMiddleware",
    "IndexRequest",
    "IndexResponse",
    "QueryRequest",
    "QueryResponse",
    "HealthResponse",
    "RepoResponse",
    "RepoListResponse",
    "CollectionResponse",
    "CollectionListResponse",
]
