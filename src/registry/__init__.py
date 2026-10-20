"""
Repository registry and incremental indexing package.
"""
from src.registry.models import (
    RepositoryMetadata,
    ChangeSet,
    IncrementalIndexResult,
)
from src.registry.registry import (
    RepoRegistry,
    get_repo_registry,
    DEFAULT_REGISTRY_PATH,
)
from src.registry.incremental import (
    IncrementalIndexer,
    detect_changes,
    index_repository_incremental,
)

__all__ = [
    "RepositoryMetadata",
    "ChangeSet",
    "IncrementalIndexResult",
    "RepoRegistry",
    "get_repo_registry",
    "DEFAULT_REGISTRY_PATH",
    "IncrementalIndexer",
    "detect_changes",
    "index_repository_incremental",
]
