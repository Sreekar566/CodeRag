"""
Pydantic data models for FastAPI request validation and structured response schemas.
"""
from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator, model_validator

# GitHub owner and repository naming constraints: alphanumeric, hyphens, underscores, dots
GITHUB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+$")
VALID_COLLECTION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


# ============================================================================
# Request Models
# ============================================================================

class IndexRequest(BaseModel):
    """Request payload for repository indexing."""
    model_config = ConfigDict(extra="ignore")

    owner: str = Field(..., min_length=1, max_length=100, description="GitHub repository owner/organization")
    repo: str = Field(..., min_length=1, max_length=100, description="GitHub repository name")
    branch: str | None = Field(default="main", description="Git branch to index (or None/'auto' to auto-detect default branch)")
    token: str | None = Field(default=None, description="GitHub personal access token (optional, falls back to GITHUB_TOKEN env var)")
    collection_name: str | None = Field(default=None, description="Custom vector store collection name")
    force: bool = Field(default=False, description="Force complete re-indexing even if commit SHA is unchanged")
    background: bool = Field(default=False, description="Run indexing as a background task and return immediately")

    @field_validator("owner")
    @classmethod
    def validate_owner(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("owner cannot be empty or whitespace")
        if "/" in cleaned:
            raise ValueError("owner must not contain slashes; specify 'owner' and 'repo' separately")
        if not GITHUB_NAME_PATTERN.match(cleaned):
            raise ValueError(
                f"owner '{v}' contains invalid characters. Must be alphanumeric, hyphens, underscores, or dots."
            )
        return cleaned

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("repo cannot be empty or whitespace")
        if "/" in cleaned:
            raise ValueError("repo must not contain slashes; specify 'owner' and 'repo' separately")
        if not GITHUB_NAME_PATTERN.match(cleaned):
            raise ValueError(
                f"repo '{v}' contains invalid characters. Must be alphanumeric, hyphens, underscores, or dots."
            )
        return cleaned

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return "main"
        return cleaned

    @field_validator("token")
    @classmethod
    def validate_token(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if not (3 <= len(cleaned) <= 63):
            raise ValueError(
                f"collection_name length must be between 3 and 63 characters, got {len(cleaned)}"
            )
        if not VALID_COLLECTION_PATTERN.match(cleaned):
            raise ValueError(
                f"collection_name '{v}' contains invalid characters. Must be alphanumeric, underscores, or hyphens."
            )
        return cleaned


class QueryRequest(BaseModel):
    """Request payload for natural language codebase queries."""
    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., min_length=1, description="Natural language question about the codebase")
    repo: str | None = Field(default=None, description="Target repository in 'owner/repo' or 'owner' format")
    repos: list[str] | None = Field(default=None, description="List of target repositories for multi-codebase search")
    collection_name: str | None = Field(default=None, description="Target vector store collection name")
    collections: list[str] | None = Field(default=None, description="List of target vector store collection names")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of candidate chunks to retrieve")
    max_context_tokens: int | None = Field(default=None, ge=64, le=32768, description="Maximum token budget for prompt context")
    rerank: bool | None = Field(default=True, description="Apply keyword, path, and definition re-ranking")
    merge_overlapping: bool | None = Field(default=True, description="Merge contiguous or overlapping chunks from the same file")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM sampling temperature")
    where: dict[str, Any] | None = Field(default=None, description="Metadata filter dictionary for vector store query")

    @field_validator("repo")
    @classmethod
    def validate_query_repo(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if "/" in cleaned:
            parts = cleaned.split("/")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(
                    f"Invalid repository specifier '{v}'. Expected 'owner/repo' or 'name'."
                )
            return f"{parts[0].strip()}/{parts[1].strip()}"
        return cleaned

    @field_validator("repos")
    @classmethod
    def validate_query_repos(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned_list: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                continue
            if "/" in cleaned:
                parts = cleaned.split("/")
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    raise ValueError(
                        f"Invalid repository specifier '{item}'. Expected 'owner/repo' or 'name'."
                    )
                cleaned = f"{parts[0].strip()}/{parts[1].strip()}"
            if cleaned not in seen:
                seen.add(cleaned)
                cleaned_list.append(cleaned)
        return cleaned_list if cleaned_list else None

    @field_validator("collection_name")
    @classmethod
    def validate_query_collection_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("collections")
    @classmethod
    def validate_query_collections(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned_list: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                cleaned_list.append(cleaned)
        return cleaned_list if cleaned_list else None


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """System health and runtime status response."""
    status: str = Field(..., description="Service status (e.g. 'ok')")
    version: str = Field(default="0.1.0", description="API version")
    timestamp: str = Field(..., description="Current UTC timestamp")
    vector_store_type: str = Field(..., description="Vector store implementation class")
    llm_provider: str = Field(..., description="Configured LLM provider name")
    llm_model: str = Field(..., description="LLM model identifier")
    indexed_repos_count: int = Field(..., description="Number of registered repositories")
    collections_count: int = Field(..., description="Number of vector store collections")


class LivenessResponse(BaseModel):
    """Kubernetes and container liveness probe response schema."""
    status: str = Field(default="alive", description="Liveness status ('alive')")
    timestamp: str = Field(..., description="Current UTC timestamp")


class SubsystemCheck(BaseModel):
    """Individual subsystem readiness check result."""
    status: str = Field(..., description="Subsystem health status ('healthy' or 'unhealthy')")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic details")
    error: str | None = Field(default=None, description="Error message if unhealthy")


class ReadinessResponse(BaseModel):
    """Kubernetes and container readiness probe response schema."""
    status: str = Field(..., description="Overall readiness status ('ready' or 'unhealthy')")
    timestamp: str = Field(..., description="Current UTC timestamp")
    checks: dict[str, SubsystemCheck] = Field(..., description="Subsystem checks for storage, registry, and persistence")


class RepoResponse(BaseModel):
    """Repository metadata response schema."""
    owner: str
    repo: str
    repo_id: str
    collection_name: str
    branch: str = "main"
    commit_sha: str | None = None
    indexed_at: str | None = None
    file_count: int = 0
    chunk_count: int = 0
    status: str = "ready"
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepoListResponse(BaseModel):
    """List of indexed repositories."""
    repos: list[RepoResponse]
    total: int


class IndexResponse(BaseModel):
    """Response returned upon completion (or scheduling) of repository indexing."""
    owner: str
    repo: str
    collection: str
    branch: str = "main"
    commit_sha: str | None = None
    status: str = Field(..., description="'created', 'updated', 'unchanged', 'indexing', or 'failed'")
    files_processed: int = 0
    chunks_stored: int = 0
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_unchanged: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    total_tree_files: int = 0
    filtered_files: int = 0
    skipped_files: list[str] = Field(default_factory=list)
    error: str | None = None
    background_task: bool = False
    message: str | None = None


class CitationModel(BaseModel):
    """Code citation referencing a file path and line span."""
    path: str
    start_line: int | None = None
    end_line: int | None = None
    repo: str | None = None
    citation_str: str


class SourceChunkModel(BaseModel):
    """Retrieved source chunk included in context."""
    id: str
    text: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None
    similarity: float | None = None
    distance: float | None = None
    repo: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Structured response from the RAG question-answering engine."""
    query: str
    answer: str
    formatted_answer: str
    citations: list[CitationModel]
    sources: list[SourceChunkModel]
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionResponse(BaseModel):
    """Vector store collection summary."""
    name: str
    chunk_count: int


class CollectionListResponse(BaseModel):
    """List of vector store collections."""
    collections: list[CollectionResponse]
    total: int


class CollectionDetailResponse(BaseModel):
    """Vector store collection details with sample chunks."""
    name: str
    chunk_count: int
    sample_chunks: list[dict[str, Any]] = Field(default_factory=list)


class ChangeSetResponse(BaseModel):
    """Summary of changes detected between repository tree and indexed state."""
    model_config = ConfigDict(extra="ignore")

    owner: str
    repo: str
    branch: str
    current_commit_sha: str | None = None
    latest_commit_sha: str | None = None
    added_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    unchanged_files: list[str] = Field(default_factory=list)
    has_changes: bool = False
    total_files: int = 0

    @model_validator(mode="before")
    @classmethod
    def remap_short_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for short_k, long_k in (
                ("added", "added_files"),
                ("modified", "modified_files"),
                ("deleted", "deleted_files"),
                ("unchanged", "unchanged_files"),
            ):
                if short_k in data and long_k not in data:
                    data[long_k] = data[short_k]
        return data

    @computed_field
    @property
    def added(self) -> list[str]:
        return self.added_files

    @computed_field
    @property
    def modified(self) -> list[str]:
        return self.modified_files

    @computed_field
    @property
    def deleted(self) -> list[str]:
        return self.deleted_files

    @computed_field
    @property
    def unchanged(self) -> list[str]:
        return self.unchanged_files


class DeleteResponse(BaseModel):
    """Operation deletion confirmation response."""
    success: bool
    message: str
    target: str


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    detail: str
    code: str | None = None


# ============================================================================
# Observability & Cost Monitoring Models
# ============================================================================

class FilteredFileResponse(BaseModel):
    """Metadata for a single file excluded during ingestion filtering."""
    path: str
    reason: str
    rule_match: str
    size_bytes: int | None = None


class IngestionAuditResponse(BaseModel):
    """Transparency audit log showing files evaluated during ingestion."""
    repo_id: str
    branch: str = "main"
    commit_sha: str | None = None
    timestamp: str
    total_scanned: int = 0
    total_kept: int = 0
    total_filtered: int = 0
    reason_counts: dict[str, int] = Field(default_factory=dict)
    kept_files: list[str] = Field(default_factory=list)
    filtered_files: list[FilteredFileResponse] = Field(default_factory=list)


class QueryTelemetryResponse(BaseModel):
    """Execution telemetry, timing breakdown, and cost for a codebase query."""
    query_id: str
    timestamp: str
    question: str
    target_repos: list[str] = Field(default_factory=list)
    model: str
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    total_duration_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    context_tokens: int | None = None
    cost_usd: float = 0.0
    retrieved_count: int = 0
    selected_count: int = 0
    citations_count: int = 0
    error: str | None = None


class MetricsSummaryResponse(BaseModel):
    """Aggregated operational telemetry and cost monitoring metrics."""
    total_queries: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_total_latency_ms: float = 0.0
    avg_stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0
