"""
Evaluation metrics computation for Codebase Intelligence RAG pipeline.
Computes deterministic retrieval rankings, citation groundedness,
hallucination rates, and symbol coverage.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from src.evaluation.models import GenerationMetrics, RetrievalMetrics


NON_CODE_DOMAIN_SUFFIXES = (
    ".com", ".org", ".net", ".edu", ".gov", ".io", ".co", ".info", ".biz"
)


def normalize_file_path(path: str) -> str:
    """Normalize file path for consistent matching across environments."""
    if not path:
        return ""
    # Strip whitespace, replace Windows backslashes, strip leading ./ and slashes
    clean = path.strip().replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    clean = clean.lstrip("/")
    return clean


def normalize_repo(repo: str) -> str:
    """Normalize repository identifier (owner/repo or repo) for comparison."""
    if not repo:
        return ""
    clean = repo.strip().replace("\\", "/").rstrip("/")
    while clean.startswith("./"):
        clean = clean[2:]
    clean = clean.lstrip("/")
    if clean.lower().endswith(".git"):
        clean = clean[:-4]
    return clean


def split_repo_and_path(path_str: str) -> tuple[str | None, str]:
    """
    Split a path string into an optional repository identifier and normalized file path.
    Handles 'repo:path', 'owner/repo:path', Windows drive letters ('C:/path'),
    and trailing line number indicators ('path:10-25').
    """
    if not path_str:
        return None, ""

    s = path_str.strip("`[]() \t\n\r\"'").replace("\\", "/")
    if not s:
        return None, ""

    # Don't split URL schemes
    if "://" in s:
        return None, s

    parts = s.split(":")
    if len(parts) == 1:
        return None, parts[0]

    # Strip trailing line numbers (e.g., :10-25 or :L10-L25) if present
    has_trailing_line = bool(re.match(r"^L?\d+(?:-L?\d+)?$", parts[-1].strip(), re.IGNORECASE))
    if has_trailing_line and len(parts) > 1:
        parts = parts[:-1]

    if len(parts) == 1:
        return None, parts[0]

    # Windows drive letter e.g. C:/path
    if len(parts[0]) == 1 and parts[0].isalpha():
        return None, ":".join(parts)

    # Domain with port e.g. example.com:8080
    if any(parts[0].lower().endswith(dom) for dom in NON_CODE_DOMAIN_SUFFIXES):
        return None, s

    # Repo with Windows drive letter e.g. repo:C:/path
    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha():
        repo = parts[0]
        path = ":".join(parts[1:])
        return repo, path

    repo = parts[0]
    path = ":".join(parts[1:])
    return repo, path


def match_repo(repo_a: str | None, repo_b: str | None) -> bool:
    """
    Check if two repository identifiers match.
    Returns True if either is None (unspecified repo matches any),
    or if both refer to the same repository (e.g. 'repo' matches 'owner/repo').
    """
    if not repo_a or not repo_b:
        return True
    ra = normalize_repo(repo_a).lower()
    rb = normalize_repo(repo_b).lower()
    if ra == rb:
        return True
    if ra.endswith("/" + rb) or rb.endswith("/" + ra):
        return True
    return False


def match_file_path(candidate: str, expected: str) -> bool:
    """
    Check if a candidate file path matches the expected file path.
    Supports repository prefixes, exact matching, and relative/suffix matching.
    """
    cand_repo, cand_path = split_repo_and_path(candidate)
    exp_repo, exp_path = split_repo_and_path(expected)

    if not match_repo(cand_repo, exp_repo):
        return False

    c = normalize_file_path(cand_path).lower()
    e = normalize_file_path(exp_path).lower()
    if not c or not e:
        return False
    if c == e:
        return True
    if c.endswith("/" + e) or e.endswith("/" + c):
        return True
    return False


def is_file_in_expected(candidate: str, expected_files: Iterable[str]) -> bool:
    """Check if candidate path matches any of the expected file paths."""
    for exp in expected_files:
        if match_file_path(candidate, exp):
            return True
    return False


def compute_retrieval_metrics(
    retrieved_chunks: list[dict[str, Any]] | list[str],
    expected_files: list[str],
    k: int = 10,
) -> RetrievalMetrics:
    """
    Compute ranking and retrieval quality metrics for a single query.

    Args:
        retrieved_chunks: List of chunk dictionaries (with 'path' key) or list of path strings.
        expected_files: Ground-truth relevant file paths.
        k: Maximum retrieval depth to evaluate.
    """
    # Extract file paths preserving rank order
    retrieved_paths: list[str] = []
    for item in retrieved_chunks:
        if isinstance(item, dict):
            meta = item.get("metadata") or {}
            p = item.get("path") or meta.get("path", "")
            repo = item.get("repo") or meta.get("repo", "")
            if repo and p and not str(p).startswith(f"{repo}:"):
                p = f"{repo}:{p}"
        else:
            p = str(item)
        if p:
            retrieved_paths.append(normalize_file_path(p))

    # Slice to top-k
    top_k_paths = retrieved_paths[:k]

    # Handle negative / out-of-scope queries (no expected files)
    if not expected_files:
        return RetrievalMetrics(
            hit=True,
            hit_rate=1.0,
            mrr=1.0,
            precision_at_k=1.0 if not top_k_paths else 0.0,
            recall_at_k=1.0,
            retrieved_files=top_k_paths,
            relevant_files_found=[],
        )

    first_rank: int | None = None
    relevant_chunks_count = 0
    found_expected_files: set[str] = set()

    for idx, path in enumerate(top_k_paths):
        is_relevant = False
        for exp in expected_files:
            if match_file_path(path, exp):
                is_relevant = True
                found_expected_files.add(exp)

        if is_relevant:
            relevant_chunks_count += 1
            if first_rank is None:
                first_rank = idx + 1  # 1-based rank

    hit = len(found_expected_files) > 0
    hit_rate = 1.0 if hit else 0.0
    mrr = (1.0 / first_rank) if first_rank is not None else 0.0
    effective_k = max(1, len(top_k_paths) if top_k_paths else k)
    precision_at_k = relevant_chunks_count / effective_k
    recall_at_k = len(found_expected_files) / max(1, len(expected_files))

    return RetrievalMetrics(
        hit=hit,
        hit_rate=hit_rate,
        mrr=round(mrr, 4),
        precision_at_k=round(precision_at_k, 4),
        recall_at_k=round(recall_at_k, 4),
        retrieved_files=top_k_paths,
        relevant_files_found=sorted(list(found_expected_files)),
    )


def compute_citation_metrics(
    citations: list[Any],
    retrieved_sources: list[dict[str, Any]],
    expected_files: list[str],
) -> tuple[float, float, list[str], float]:
    """
    Evaluate citation precision, citation recall, hallucinated citations, and citation groundedness.

    Returns:
        (citation_precision, citation_recall, hallucinated_citations, groundedness_score)
    """
    # Extract source paths available in the context chunks
    context_paths: list[str] = []
    for s in retrieved_sources:
        if isinstance(s, dict):
            meta = s.get("metadata") or {}
            p = s.get("path") or meta.get("path", "")
            repo = s.get("repo") or meta.get("repo", "")
            if p:
                if repo and not str(p).startswith(f"{repo}:"):
                    context_paths.append(f"{repo}:{p}")
                else:
                    context_paths.append(str(p))
        elif s:
            context_paths.append(str(s))

    # Extract cited paths
    cited_paths: list[str] = []
    for c in citations:
        if hasattr(c, "path"):
            p = str(c.path)
            repo = getattr(c, "repo", None)
            if repo and not p.startswith(f"{repo}:"):
                cited_paths.append(f"{repo}:{p}")
            else:
                cited_paths.append(p)
        elif isinstance(c, dict) and "path" in c:
            p = str(c["path"])
            repo = c.get("repo")
            if repo and not p.startswith(f"{repo}:"):
                cited_paths.append(f"{repo}:{p}")
            else:
                cited_paths.append(p)
        elif isinstance(c, str):
            clean_str = c.strip("`[]() \t\n\r\"'")
            repo, path = split_repo_and_path(clean_str)
            if repo and path and not path.startswith(f"{repo}:"):
                cited_paths.append(f"{repo}:{path}")
            elif path:
                cited_paths.append(path)
            elif clean_str:
                cited_paths.append(clean_str)

    if not cited_paths:
        # If no citations were made:
        if not expected_files:
            return 1.0, 1.0, [], 1.0
        return 0.0, 0.0, [], 1.0

    # Check for hallucinated citations (citations whose file is not in retrieved sources)
    hallucinated: list[str] = []
    grounded_count = 0
    for cp in cited_paths:
        matched = False
        for ctx_p in context_paths:
            if match_file_path(cp, ctx_p):
                matched = True
                break
        if matched:
            grounded_count += 1
        else:
            hallucinated.append(cp)

    groundedness = grounded_count / len(cited_paths) if cited_paths else 1.0

    # Compute citation precision against ground truth expected files
    if expected_files:
        relevant_citations = 0
        for cp in cited_paths:
            if is_file_in_expected(cp, expected_files):
                relevant_citations += 1
        citation_precision = relevant_citations / len(cited_paths)

        # Compute citation recall (fraction of expected files cited)
        cited_expected = set()
        for exp in expected_files:
            for cp in cited_paths:
                if match_file_path(cp, exp):
                    cited_expected.add(exp)
        citation_recall = len(cited_expected) / len(expected_files)
    else:
        # For negative queries, any citation of an expected file that doesn't exist is 0 precision
        citation_precision = 1.0 if not cited_paths else 0.0
        citation_recall = 1.0

    return (
        round(citation_precision, 4),
        round(citation_recall, 4),
        hallucinated,
        round(groundedness, 4),
    )


def compute_symbol_coverage(
    answer: str,
    expected_symbols: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Check for the presence of key symbols (classes, functions, variables) in the answer.

    Returns:
        (coverage_score, found_symbols, missing_symbols)
    """
    if not expected_symbols:
        return 1.0, [], []

    if not answer:
        return 0.0, [], list(expected_symbols)

    found: list[str] = []
    missing: list[str] = []
    answer_lower = answer.lower()

    for sym in expected_symbols:
        sym_clean = sym.strip()
        # Word-boundary or token search
        pattern = r"\b" + re.escape(sym_clean.lower()) + r"\b"
        if re.search(pattern, answer_lower):
            found.append(sym_clean)
        else:
            missing.append(sym_clean)

    coverage = len(found) / len(expected_symbols)
    return round(coverage, 4), found, missing


ABSTENTION_PHRASES = [
    "not found",
    "not implemented",
    "does not appear",
    "does not exist",
    "no mention",
    "not present",
    "not configured",
    "no references",
    "cannot find",
    "could not find",
    "not included",
    "no evidence",
    "not supported",
    "is not used",
    "is not part of",
    "none found",
    "no payment",
    "no stripe",
    "no helm",
    "no kubernetes",
    "not in the codebase",
]


def evaluate_abstention(answer: str) -> tuple[bool, bool]:
    """
    Evaluate whether the model correctly abstained from answering an unanswerable/negative query.

    Returns:
        (abstained_correctly, hallucinated_feature)
    """
    if not answer:
        return True, False

    answer_lower = answer.lower()
    for phrase in ABSTENTION_PHRASES:
        if phrase in answer_lower:
            return True, False

    return False, True
