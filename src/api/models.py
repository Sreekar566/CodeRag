"""
Pydantic data models for FastAPI request validation and structured response schemas.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, ConfigDict


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


class DeleteResponse(BaseModel):
    """Operation deletion confirmation response."""
    success: bool
    message: str
    target: str


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    detail: str
    code: str | None = None
