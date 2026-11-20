"""
Code chunking utilities for splitting source code files into overlapping windows.
Snaps boundaries to nearby blank lines and preserves 1-based line range metadata.
"""
from __future__ import annotations

import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any

# Default chunk size and overlap in approximate tokens (whitespace-separated words)
CHUNK_SIZE = 400
OVERLAP = 50

# How far (in lines) to search outward for a nearby blank line when snapping chunk boundaries
BOUNDARY_SEARCH_WINDOW = 5

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".less": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".r": "r",
    ".lua": "lua",
    ".xml": "xml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".txt": "text",
}

FILENAME_TO_LANGUAGE: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "vagrantfile": "ruby",
    "procfile": "yaml",
}


def detect_language(path: str) -> str:
    """
    Detect the programming or markup language from a file path based on
    its filename or extension.
    """
    filename = os.path.basename(path).lower()
    if filename in FILENAME_TO_LANGUAGE:
        return FILENAME_TO_LANGUAGE[filename]

    _, ext = os.path.splitext(filename)
    if ext in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext]

    return "unknown"


def compute_cumulative_words(lines: list[str]) -> list[int]:
    """
    Compute prefix sum of word counts across lines.
    cumulative_words[i] represents the total words before line i (in lines[0:i]).
    The final element is a sentinel representing total words in the entire file.
    """
    cumulative = [0]
    total = 0
    for line in lines:
        total += len(line.split())
        cumulative.append(total)
    return cumulative


def find_line_index_for_word(cumulative_words: list[int], target_word: int) -> int:
    """
    Find the line index corresponding to a word offset using binary search.
    Returns the first line index where cumulative_words[i] >= target_word.
    Runs in O(log L) time.
    """
    return bisect_left(cumulative_words, target_word)


def find_start_line_index_for_word(cumulative_words: list[int], target_word: int) -> int:
    """
    Find the 0-based line index containing the word at target_word offset.
    Returns line index i such that cumulative_words[i] <= target_word < cumulative_words[i+1].
    Runs in O(log L) time.
    """
    if target_word <= 0:
        return 0
    idx = bisect_right(cumulative_words, target_word) - 1
    return max(0, idx)


def snap_boundary_to_blank_line(
    lines: list[str],
    end_line: int,
    start_line: int,
    window: int = BOUNDARY_SEARCH_WINDOW,
) -> int:
    """
    Search outward from end_line within a window to find the closest blank line.
    Prevents snapping if it would collapse the chunk to or below start_line.
    """
    if window is None or window <= 0:
        return end_line

    for offset in range(window + 1):
        before = end_line - offset
        after = end_line + offset

        if 0 <= before < len(lines) and before > start_line and lines[before].strip() == "":
            return before
        if 0 <= after < len(lines) and after > start_line and lines[after].strip() == "":
            return after

    return end_line


def build_chunk(
    path: str,
    lines: list[str],
    start_line: int,
    end_line: int,
    language: str | None = None,
) -> dict[str, Any]:
    """
    Construct a standardized chunk dictionary from line boundaries.
    Converts 0-based start_line index to 1-based start line number.
    Ensures end_line is at least start_line + 1 to avoid empty slices and inverted line ranges.
    """
    safe_start = max(0, start_line)
    safe_end = max(safe_start + 1, end_line)
    if lines:
        safe_end = min(len(lines), safe_end)
        safe_start = min(safe_start, safe_end - 1)
    chunk_body = "\n".join(lines[safe_start:safe_end])
    return {
        "path": path,
        "start_line": safe_start + 1,  # 1-based indexing for citations
        "end_line": max(safe_start + 1, safe_end),
        "text": chunk_body,
        "language": language or detect_language(path),
    }


@dataclass
class CodeChunker:
    """
    Configurable chunker for source code files using word count windows
    and blank-line boundary snapping.
    """
    chunk_size: int = CHUNK_SIZE
    overlap: int = OVERLAP
    boundary_search_window: int = BOUNDARY_SEARCH_WINDOW

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        if self.boundary_search_window is not None and self.boundary_search_window < 0:
            raise ValueError(
                f"boundary_search_window must be non-negative or None, got {self.boundary_search_window}"
            )

    def chunk(
        self,
        text: str,
        path: str,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Split a file's text into overlapping chunks, each tagged with the file
        path, language, and 1-based line range.
        """
        if not text or not text.strip():
            return []

        lines = text.split("\n")
        cumulative_words = compute_cumulative_words(lines)
        total_words = cumulative_words[-1]

        if total_words == 0:
            return []

        step = self.chunk_size - self.overlap
        detected_lang = language or detect_language(path)

        chunks: list[dict[str, Any]] = []
        start_word = 0

        while start_word < total_words:
            end_word = min(start_word + self.chunk_size, total_words)

            start_line = find_start_line_index_for_word(cumulative_words, start_word)
            end_line = find_line_index_for_word(cumulative_words, end_word)

            if end_line <= start_line:
                end_line = min(len(lines), start_line + 1)

            # Snap end boundary to nearest blank line if configured and not final chunk
            if self.boundary_search_window and end_word < total_words:
                end_line = snap_boundary_to_blank_line(
                    lines=lines,
                    end_line=end_line,
                    start_line=start_line,
                    window=self.boundary_search_window,
                )
                if end_line <= start_line:
                    end_line = min(len(lines), start_line + 1)

            chunk = build_chunk(
                path=path,
                lines=lines,
                start_line=start_line,
                end_line=end_line,
                language=detected_lang,
            )

            # Avoid adding empty chunks or exact duplicates
            if chunk["text"].strip():
                if not chunks or (
                    chunk["start_line"] != chunks[-1]["start_line"]
                    or chunk["end_line"] != chunks[-1]["end_line"]
                ):
                    chunks.append(chunk)

            # If this chunk already covered up to or beyond the end of the file, we are done
            if end_line >= len(lines):
                break

            start_word += step

        return chunks


def chunk_text(
    text: str,
    path: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    boundary_window: int = BOUNDARY_SEARCH_WINDOW,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """
    Splits a file's text into overlapping chunks, each tagged with the file
    path, language, and the line range it covers.

    Returns a list of dicts like:
        {"path": "app/models.py", "start_line": 1, "end_line": 42, "text": "...", "language": "python"}
    """
    chunker = CodeChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        boundary_search_window=boundary_window,
    )
    return chunker.chunk(text, path, language=language)
