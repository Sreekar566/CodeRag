"""
File filtering logic for codebase ingestion.
Implements blocklist filtering for noise directories, noise extensions/binaries,
and oversized files, alongside transparency audit logging.
"""
from __future__ import annotations

from typing import Any

# Noise directories (folder names that are never authored source)
NOISE_DIRS = {
    "migrations",
    "node_modules",
    "__pycache__",
    ".git",
    "venv",
    ".venv",
    ".pytest_cache",
    "dist",
    "build",
}

# Noise extensions (file endings and metadata files that are not authored source)
NOISE_EXTENSIONS = {
    ".gitignore",
    ".gitattributes",
    ".flaskenv",
    ".gif",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".ico",
    ".svg",
    ".mp4",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".avi",
    ".mov",
    ".wmv",
    ".mkv",
    ".zip",
    ".tar.gz",
    ".rar",
    ".7z",
    ".po",
    "LICENSE",
}

# Size threshold in bytes
MAX_FILE_SIZE = 500_000


def classify_file(file_info: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """
    Classify whether a file should be kept or filtered out.
    Returns: (is_kept, reason, rule_match)
    """
    path = file_info.get("path", "")
    parts = path.split("/")

    # 1. Noise directory check
    for dir_name in NOISE_DIRS:
        if dir_name in parts:
            return False, "noise_directory", dir_name

    # 2. Noise extension / filename check
    for ext in NOISE_EXTENSIONS:
        if path.endswith(ext):
            return False, "noise_extension", ext

    # 3. File size check
    size = file_info.get("size")
    if MAX_FILE_SIZE is not None and size is not None and size > MAX_FILE_SIZE:
        return False, "file_too_large", f"size {size} > {MAX_FILE_SIZE}"

    return True, None, None


def audit_filter_files(
    tree: list[dict[str, Any]],
    repo_id: str = "unknown",
    branch: str = "main",
    commit_sha: str | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """
    Takes the list of file dicts from get_repo_tree(), filters noise files,
    and returns a tuple of (kept_files, IngestionAuditLog).
    """
    from src.observability.models import FilteredFileRecord, IngestionAuditLog

    kept: list[dict[str, Any]] = []
    filtered_records: list[FilteredFileRecord] = []
    reason_counts: dict[str, int] = {}
    kept_paths: list[str] = []

    for f in tree:
        is_kept, reason, rule_match = classify_file(f)
        if is_kept:
            kept.append(f)
            kept_paths.append(f.get("path", ""))
        else:
            reason_str = reason or "unknown"
            rule_str = rule_match or ""
            reason_counts[reason_str] = reason_counts.get(reason_str, 0) + 1
            filtered_records.append(
                FilteredFileRecord(
                    path=f.get("path", ""),
                    reason=reason_str,
                    rule_match=rule_str,
                    size_bytes=f.get("size"),
                )
            )

    audit_log = IngestionAuditLog(
        repo_id=repo_id,
        branch=branch,
        commit_sha=commit_sha,
        total_scanned=len(tree),
        total_kept=len(kept),
        total_filtered=len(filtered_records),
        reason_counts=reason_counts,
        kept_files=kept_paths,
        filtered_files=filtered_records,
    )

    return kept, audit_log


def filter_files(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Takes the list of file dicts from get_repo_tree(), returns only the ones
    worth chunking/embedding. Maintains backwards compatibility.
    """
    kept, _ = audit_filter_files(tree)
    return kept
