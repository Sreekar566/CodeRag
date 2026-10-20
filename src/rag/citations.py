"""
Citation data structures and extraction utilities for grounding RAG answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Citation:
    """Represents a code source citation with file path and line numbers."""
    path: str
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None
    snippet: str | None = None

    def format_citation(self) -> str:
        """Format as path:start-end or path:start or path."""
        if self.start_line is not None and self.end_line is not None:
            if self.start_line == self.end_line:
                return f"{self.path}:{self.start_line}"
            return f"{self.path}:{self.start_line}-{self.end_line}"
        elif self.start_line is not None:
            return f"{self.path}:{self.start_line}"
        return self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "formatted": self.format_citation(),
            "snippet": self.snippet,
        }

    def __str__(self) -> str:
        return self.format_citation()


# Regex pattern to match path:line(-line)? references in LLM text
# Examples:
# - app/auth/routes.py:10-25
# - app/auth/routes.py:12
# - `src/storage/base.py:1-40`
# - [models.py:L15-L30]
CITATION_REGEX = re.compile(
    r"(?:`|\[)?([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+|[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)+)"
    r":(?:L)?(\d+)(?:-(?:L)?(\d+))?(?:`|\])?",
    re.IGNORECASE,
)


NON_CODE_DOMAIN_SUFFIXES = (
    ".com", ".org", ".net", ".edu", ".gov", ".io", ".co", ".info", ".biz"
)


def extract_citations_from_text(text: str) -> list[Citation]:
    """
    Parse citations from markdown or plain text responses.
    Filters out URLs and non-code references.
    """
    if not text:
        return []

    citations: list[Citation] = []
    seen: set[tuple[str, int | None, int | None]] = set()

    for match in CITATION_REGEX.finditer(text):
        # Check context immediately preceding match for URL scheme
        start_idx = match.start()
        prefix = text[max(0, start_idx - 10):start_idx]
        if "://" in prefix or prefix.endswith("//"):
            continue

        raw_path = match.group(1).strip()
        # Skip URL schemes or protocol prefixes or double slash prefixes
        if raw_path.lower() in ("http", "https", "ftp", "file") or raw_path.startswith("//"):
            continue

        # Skip web domain addresses like example.com:8080
        if any(raw_path.lower().endswith(dom) for dom in NON_CODE_DOMAIN_SUFFIXES):
            continue

        raw_path = raw_path.lstrip("./")

        start_str = match.group(2)
        end_str = match.group(3)

        start_line = int(start_str) if start_str is not None else None
        end_line = int(end_str) if end_str is not None else start_line

        key = (raw_path, start_line, end_line)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    path=raw_path,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

    return citations


def citations_from_chunks(chunks: list[dict[str, Any]]) -> list[Citation]:
    """
    Generate citations directly from retrieved source chunks.
    """
    citations: list[Citation] = []
    seen: set[tuple[str, Any, Any]] = set()

    for chunk in chunks:
        path = chunk.get("path") or chunk.get("metadata", {}).get("path")
        if not path:
            continue

        start = chunk.get("start_line") or chunk.get("metadata", {}).get("start_line")
        end = chunk.get("end_line") or chunk.get("metadata", {}).get("end_line")
        lang = chunk.get("language") or chunk.get("metadata", {}).get("language")
        snippet = chunk.get("text")

        key = (path, start, end)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    path=str(path),
                    start_line=int(start) if start is not None else None,
                    end_line=int(end) if end is not None else None,
                    language=str(lang) if lang else None,
                    snippet=snippet,
                )
            )

    return citations
