"""
Ingestion package for codebase filtering, chunking, and repository retrieval.
"""
from src.ingestion.chunker import (
    CodeChunker,
    chunk_text,
    compute_cumulative_words,
    find_line_index_for_word,
    snap_boundary_to_blank_line,
    build_chunk,
    detect_language,
    CHUNK_SIZE,
    OVERLAP,
    BOUNDARY_SEARCH_WINDOW,
)
from src.ingestion.filters import (
    filter_files,
    classify_file,
    audit_filter_files,
    NOISE_DIRS,
    NOISE_EXTENSIONS,
    MAX_FILE_SIZE,
)
from src.ingestion.github_client import (
    get_repo_tree,
    get_file_content,
    get_default_branch,
    get_latest_commit_sha,
)

__all__ = [
    # Chunker
    "CodeChunker",
    "chunk_text",
    "compute_cumulative_words",
    "find_line_index_for_word",
    "snap_boundary_to_blank_line",
    "build_chunk",
    "detect_language",
    "CHUNK_SIZE",
    "OVERLAP",
    "BOUNDARY_SEARCH_WINDOW",
    # Filters
    "filter_files",
    "classify_file",
    "audit_filter_files",
    "NOISE_DIRS",
    "NOISE_EXTENSIONS",
    "MAX_FILE_SIZE",
    # GitHub Client
    "get_repo_tree",
    "get_file_content",
    "get_default_branch",
    "get_latest_commit_sha",
]
