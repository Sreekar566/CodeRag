"""
Re-ranking, score adjustment, and chunk deduplication/merging for codebase RAG.
Combines semantic similarity with lexical keywords, file path relevance, and symbol definitions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "but", "not", "what", "which", "who", "whom", "this", "that", "these",
    "those", "how", "where", "when", "why", "can", "could", "will", "would",
    "should", "code", "repo", "repository", "file", "function", "class", "method",
}


def extract_query_keywords(query: str) -> list[str]:
    """
    Extract meaningful identifiers and keywords from a user question.
    Includes snake_case, camelCase split tokens, and original identifiers.
    """
    if not query:
        return []

    # Find words and tokens
    raw_tokens = re.findall(r"[A-Za-z0-9_]+", query)
    keywords: set[str] = set()

    for token in raw_tokens:
        token_lower = token.lower()
        if len(token) >= 2 and token_lower not in STOPWORDS:
            keywords.add(token_lower)

        # Split camelCase (e.g. getUserInfo -> get, user, info)
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[0-9]+", token)
        if len(camel_parts) > 1:
            for part in camel_parts:
                part_lower = part.lower()
                if len(part) >= 2 and part_lower not in STOPWORDS:
                    keywords.add(part_lower)

        # Split snake_case (e.g. get_user_info -> get, user, info)
        if "_" in token:
            for part in token.split("_"):
                part_lower = part.lower()
                if len(part) >= 2 and part_lower not in STOPWORDS:
                    keywords.add(part_lower)

    return sorted(keywords)


def compute_lexical_score(text: str, keywords: list[str]) -> float:
    """
    Compute lexical match score (0.0 to 1.0) based on keyword frequency and presence in text.
    """
    if not text or not keywords:
        return 0.0

    text_lower = text.lower()
    matched = sum(1 for kw in keywords if kw in text_lower)
    return matched / len(keywords)


def compute_path_score(path: str, keywords: list[str]) -> float:
    """
    Compute relevance score (0.0 to 1.0) based on query keywords matching file path segments.
    """
    if not path or not keywords:
        return 0.0

    path_lower = path.lower()
    matched = sum(1 for kw in keywords if kw in path_lower)
    return matched / len(keywords)


def compute_definition_score(text: str, keywords: list[str]) -> float:
    """
    Detect if text contains class, function, or variable definitions matching query keywords.
    """
    if not text or not keywords:
        return 0.0

    text_lower = text.lower()
    for kw in keywords:
        # Regex patterns for def/class/function definitions across Python, JS/TS, Go, Rust, etc.
        patterns = [
            rf"\bdef\s+{kw}\b",
            rf"\bclass\s+{kw}\b",
            rf"\bfunction\s+{kw}\b",
            rf"\bfn\s+{kw}\b",
            rf"\bconst\s+{kw}\s*=",
            rf"\blet\s+{kw}\s*=",
            rf"\bvar\s+{kw}\s*=",
        ]
        if any(re.search(pat, text_lower) for pat in patterns):
            return 1.0

    return 0.0


def _extract_line_int(chunk: dict[str, Any], key: str) -> int | None:
    val = chunk.get(key)
    if val is None and isinstance(chunk.get("metadata"), dict):
        val = chunk["metadata"].get(key)
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    return None


def _extract_score(chunk: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = chunk.get(key)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return default


def merge_overlapping_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge contiguous or overlapping chunks from the exact same file.
    Preserves line order and updates start_line, end_line, and combined text.
    Safely handles missing or None line numbers, None texts, and None similarity scores.
    """
    if not chunks or len(chunks) <= 1:
        return [c.copy() for c in chunks] if chunks else []

    # Group by repository and file path to avoid cross-repo chunk collisions
    by_file: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    for c in chunks:
        path = c.get("path") or (c.get("metadata", {}) or {}).get("path") or "unknown"
        repo = c.get("repo") or (c.get("metadata", {}) or {}).get("repo")
        by_file.setdefault((repo, path), []).append(c)

    merged_all: list[dict[str, Any]] = []

    for (repo, path), file_chunks in by_file.items():
        # Sort primarily by start_line, secondarily by end_line.
        # Chunks with None start_line are placed first (-1) without raising TypeError.
        file_chunks.sort(key=lambda c: (
            _extract_line_int(c, "start_line") if _extract_line_int(c, "start_line") is not None else -1,
            _extract_line_int(c, "end_line") if _extract_line_int(c, "end_line") is not None else -1,
        ))

        current = file_chunks[0].copy()
        if isinstance(current.get("metadata"), dict):
            current["metadata"] = current["metadata"].copy()
        current_text = current.get("text") or ""
        current_lines = current_text.splitlines()
        current_start = _extract_line_int(current, "start_line")
        current_end = _extract_line_int(current, "end_line")

        for nxt in file_chunks[1:]:
            nxt_start = _extract_line_int(nxt, "start_line")
            nxt_end = _extract_line_int(nxt, "end_line")
            nxt_text = nxt.get("text") or ""
            nxt_lines = nxt_text.splitlines()

            # If either chunk lacks valid line numbers, we cannot safely merge by line overlap
            if current_start is None or current_end is None or nxt_start is None or nxt_end is None:
                if current_start is not None:
                    current["start_line"] = current_start
                if current_end is not None:
                    current["end_line"] = current_end
                current["text"] = "\n".join(current_lines) if current_lines else current_text
                merged_all.append(current)

                current = nxt.copy()
                if isinstance(current.get("metadata"), dict):
                    current["metadata"] = current["metadata"].copy()
                current_text = nxt_text
                current_lines = nxt_lines
                current_start = nxt_start
                current_end = nxt_end
                continue

            # Check if overlapping or touching (start <= end + 1)
            if nxt_start <= current_end + 1:
                # Overlapping or adjacent: preserve highest relevance score
                c_sim = _extract_score(current, "similarity", _extract_score(current, "score", 0.0))
                n_sim = _extract_score(nxt, "similarity", _extract_score(nxt, "score", 0.0))
                current["similarity"] = max(c_sim, n_sim)
                if "score" in current or "score" in nxt:
                    c_sc = _extract_score(current, "score", 0.0)
                    n_sc = _extract_score(nxt, "score", 0.0)
                    current["score"] = max(c_sc, n_sc)

                if nxt_end > current_end:
                    # Calculate line overlap
                    overlap_lines = current_end - nxt_start + 1
                    if 0 <= overlap_lines < len(nxt_lines):
                        new_lines = nxt_lines[overlap_lines:]
                    else:
                        # Fallback if line count doesn't match line numbers
                        new_lines = [l for l in nxt_lines if l not in current_lines]

                    current_lines.extend(new_lines)
                    current_end = nxt_end
                    current_text = "\n".join(current_lines)
                    current["text"] = current_text
                    current["end_line"] = current_end
                    if "metadata" in current and isinstance(current["metadata"], dict):
                        current["metadata"]["end_line"] = current_end

                    if repo:
                        current["id"] = f"{repo}:{path}:{current_start}-{current_end}"
                    else:
                        current["id"] = f"{path}:{current_start}-{current_end}"
            else:
                # Gap between chunks; finalize current and start next
                current["start_line"] = current_start
                current["end_line"] = current_end
                current["text"] = "\n".join(current_lines) if current_lines else current_text
                merged_all.append(current)

                current = nxt.copy()
                if isinstance(current.get("metadata"), dict):
                    current["metadata"] = current["metadata"].copy()
                current_text = nxt_text
                current_lines = nxt_lines
                current_start = nxt_start
                current_end = nxt_end

        if current_start is not None:
            current["start_line"] = current_start
        if current_end is not None:
            current["end_line"] = current_end
        current["text"] = "\n".join(current_lines) if current_lines else current_text
        merged_all.append(current)

    return merged_all


@dataclass
class ReRanker:
    """
    Code-aware re-ranking engine combining semantic similarity with lexical,
    path-level, and symbol definition signals.
    """
    semantic_weight: float = 0.50
    lexical_weight: float = 0.25
    path_weight: float = 0.15
    definition_weight: float = 0.10

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        merge_overlapping: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Re-rank code chunks based on multi-factor relevance to query.
        """
        if not chunks:
            return []

        candidates = merge_overlapping_chunks(chunks) if merge_overlapping else [c.copy() for c in chunks]
        keywords = extract_query_keywords(query)

        reranked: list[dict[str, Any]] = []
        for chunk in candidates:
            item = chunk.copy()
            text = item.get("text", "")
            path = item.get("path") or item.get("metadata", {}).get("path", "")

            # Semantic score (normalized 0 to 1)
            raw_sim = item.get("similarity", item.get("score", 0.5))
            try:
                semantic_score = float(raw_sim)
            except (ValueError, TypeError):
                semantic_score = 0.5
            # Clamp between 0.0 and 1.0
            semantic_score = max(0.0, min(1.0, semantic_score))

            # Feature scores
            lexical_score = compute_lexical_score(text, keywords)
            path_score = compute_path_score(path, keywords)
            definition_score = compute_definition_score(text, keywords)

            # Combined score
            final_score = (
                self.semantic_weight * semantic_score
                + self.lexical_weight * lexical_score
                + self.path_weight * path_score
                + self.definition_weight * definition_score
            )

            item["rerank_score"] = round(final_score, 4)
            item["lexical_score"] = round(lexical_score, 4)
            item["path_score"] = round(path_score, 4)
            item["definition_score"] = round(definition_score, 4)
            reranked.append(item)

        # Sort descending by final score
        reranked.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return reranked


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    merge_overlapping: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Convenience function to re-rank code chunks for a query.
    """
    reranker = ReRanker(**kwargs)
    return reranker.rerank(query, chunks, merge_overlapping=merge_overlapping)
