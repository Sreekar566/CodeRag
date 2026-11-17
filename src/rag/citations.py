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
    repo: str | None = None

    def format_citation(self) -> str:
        """Format as [repo:]path:start-end or [repo:]path:start or [repo:]path."""
        if self.start_line is not None and self.end_line is not None:
            if self.start_line == self.end_line:
                loc = f"{self.path}:{self.start_line}"
            else:
                loc = f"{self.path}:{self.start_line}-{self.end_line}"
        elif self.start_line is not None:
            loc = f"{self.path}:{self.start_line}"
        else:
            loc = self.path

        if self.repo and not self.path.startswith(f"{self.repo}/"):
            return f"{self.repo}:{loc}"
        return loc

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "formatted": self.format_citation(),
            "snippet": self.snippet,
            "repo": self.repo,
        }

    def __str__(self) -> str:
        return self.format_citation()


# Regex pattern to match [repo:]path:line(-line)? references in LLM text
# Examples:
# - app/auth/routes.py:10-25
# - app/auth/routes.py:12
# - `src/storage/base.py:1-40`
# - [models.py:L15-L30]
# - myorg/repo:app/auth/routes.py:10-25
# - `backend:app/routes.py:1-20`
CITATION_REGEX = re.compile(
    r"(?:`|\[)?"
    r"(?:([a-zA-Z0-9_.\-]+(?:/[a-zA-Z0-9_.\-]+)?):)?"
    r"([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+|[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)+)"
    r":(?:L)?(\d+)(?:-(?:L)?(\d+))?"
    r"(?:`|\])?",
    re.IGNORECASE,
)


NON_CODE_DOMAIN_SUFFIXES = (
    ".com", ".org", ".net", ".edu", ".gov", ".io", ".co", ".info", ".biz"
)


def extract_citations_from_text(text: str) -> list[Citation]:
    """
    Parse citations from markdown or plain text responses.
    Supports optional repository prefixes ([repo:]path:start-end) and line ranges.
    Filters out URLs and non-code references.
    """
    if not text:
        return []

    citations: list[Citation] = []
    seen: set[tuple[str | None, str, int | None, int | None]] = set()

    for match in CITATION_REGEX.finditer(text):
        # Check context immediately preceding match for URL scheme
        start_idx = match.start()
        prefix = text[max(0, start_idx - 10):start_idx]
        if "://" in prefix or prefix.endswith("//"):
            continue

        raw_repo = match.group(1)
        raw_path = match.group(2).strip()

        # Skip URL schemes or protocol prefixes or double slash prefixes
        if raw_repo and (raw_repo.lower() in ("http", "https", "ftp", "file") or "://" in raw_repo or raw_repo.startswith("//")):
            continue
        if raw_path.lower() in ("http", "https", "ftp", "file") or raw_path.startswith("//") or "://" in raw_path:
            continue

        # Skip web domain addresses like example.com:8080
        if any(raw_path.lower().endswith(dom) for dom in NON_CODE_DOMAIN_SUFFIXES):
            continue
        if raw_repo and any(raw_repo.lower().endswith(dom) for dom in NON_CODE_DOMAIN_SUFFIXES):
            continue

        # Disambiguate single-letter Windows drive letters like C:
        repo = raw_repo.strip() if raw_repo else None
        if repo and len(repo) == 1 and repo.isalpha():
            raw_path = f"{repo}:{raw_path}"
            repo = None

        if repo:
            if repo.startswith("./"):
                repo = repo[2:]
            elif repo.startswith("/"):
                repo = repo[1:]

        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        elif raw_path.startswith("/"):
            raw_path = raw_path[1:]

        start_str = match.group(3)
        end_str = match.group(4)

        start_line = int(start_str) if start_str is not None else None
        end_line = int(end_str) if end_str is not None else start_line

        if start_line is not None and end_line is not None and start_line > end_line:
            start_line, end_line = end_line, start_line

        key = (repo, raw_path, start_line, end_line)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    path=raw_path,
                    start_line=start_line,
                    end_line=end_line,
                    repo=repo,
                )
            )

    return citations


def citations_from_chunks(chunks: list[dict[str, Any]]) -> list[Citation]:
    """
    Generate citations directly from retrieved source chunks.
    """
    citations: list[Citation] = []
    seen: set[tuple[str | None, str, Any, Any]] = set()

    for chunk in chunks:
        path = chunk.get("path") or (chunk.get("metadata", {}) or {}).get("path")
        if not path:
            continue

        repo = chunk.get("repo") or (chunk.get("metadata", {}) or {}).get("repo")
        start = chunk.get("start_line") or (chunk.get("metadata", {}) or {}).get("start_line")
        end = chunk.get("end_line") or (chunk.get("metadata", {}) or {}).get("end_line")
        lang = chunk.get("language") or (chunk.get("metadata", {}) or {}).get("language")
        snippet = chunk.get("text")

        start_val = int(start) if start is not None else None
        end_val = int(end) if end is not None else start_val
        if start_val is not None and end_val is not None and start_val > end_val:
            start_val, end_val = end_val, start_val

        key = (repo, str(path), start_val, end_val)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    path=str(path),
                    start_line=start_val,
                    end_line=end_val,
                    language=str(lang) if lang else None,
                    snippet=snippet,
                    repo=repo,
                )
            )

    return citations
