"""
API route definitions.
"""
from src.api.routes.health import router as health_router
from src.api.routes.repos import router as repos_router
from src.api.routes.index import router as index_router
from src.api.routes.query import router as query_router
from src.api.routes.collections import router as collections_router

__all__ = [
    "health_router",
    "repos_router",
    "index_router",
    "query_router",
    "collections_router",
]
