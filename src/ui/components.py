"""
UI presentation components and formatting helpers for Streamlit.
"""
from __future__ import annotations

from typing import Any
import streamlit as st


def format_citation(citation: dict[str, Any] | Any) -> str:
    """
    Format a citation object into a clear code reference string.
    """
    if isinstance(citation, str):
        return citation

    if isinstance(citation, dict):
        if citation.get("citation_str"):
            return str(citation["citation_str"])

        repo = citation.get("repo")
        path = citation.get("path", "unknown")
        start = citation.get("start_line")
        end = citation.get("end_line")

        prefix = f"{repo}:" if repo else ""
        if start is not None and end is not None:
            return f"{prefix}{path}:{start}-{end}"
        if start is not None:
            return f"{prefix}{path}:{start}"
        return f"{prefix}{path}"

    return str(citation)


def format_metrics(metadata: dict[str, Any] | None) -> str:
    """
    Format RAG query execution metrics into a concise human-readable summary.
    """
    if not metadata:
        return ""

    parts = []
    tokens = metadata.get("context_tokens")
    max_tokens = metadata.get("max_context_tokens")
    if tokens is not None:
        if max_tokens:
            parts.append(f"Tokens: {tokens}/{max_tokens}")
        else:
            parts.append(f"Tokens: {tokens}")

    retrieved = metadata.get("retrieved_count")
    selected = metadata.get("selected_count")
    if retrieved is not None:
        if selected is not None:
            parts.append(f"Chunks: {selected}/{retrieved}")
        else:
            parts.append(f"Chunks: {retrieved}")

    if metadata.get("reranked"):
        parts.append("Re-ranked: Yes")

    elapsed = metadata.get("elapsed_seconds")
    if elapsed is not None:
        parts.append(f"Time: {elapsed:.2f}s")

    return " | ".join(parts) if parts else ""


def format_timestamp(ts: str | None) -> str:
    """
    Format ISO timestamp into a clean display date.
    """
    if not ts:
        return "N/A"
    try:
        # Simplifies '2026-10-21T00:00:00.000000' -> '2026-10-21 00:00:00 UTC'
        clean = ts.replace("T", " ").split(".")[0]
        return f"{clean} UTC"
    except Exception:
        return str(ts)


def get_status_badge(status: str | None) -> tuple[str, str]:
    """
    Return an emoji icon and friendly label for an operational status.
    """
    if not status:
        return ("⚪", "Unknown")

    s = status.lower()
    if s in ("ready", "indexed", "ok", "created", "updated"):
        return ("🟢", s.capitalize())
    elif s in ("indexing", "pending", "running"):
        return ("🟡", s.capitalize())
    elif s in ("unchanged", "skipped"):
        return ("⚪", s.capitalize())
    elif s in ("failed", "error"):
        return ("🔴", s.capitalize())
    return ("ℹ️", s.capitalize())


def format_repo_label(repo: dict[str, Any]) -> str:
    """
    Format repository metadata into an informative selector label.
    """
    owner = repo.get("owner", "")
    name = repo.get("repo", "")
    chunks = repo.get("chunk_count", 0)
    files = repo.get("file_count", 0)
    commit = (repo.get("commit_sha") or "")[:7]
    sha_str = f" @ {commit}" if commit else ""
    return f"{owner}/{name} ({files} files, {chunks} chunks{sha_str})"


def format_changes_summary(changes: dict[str, Any]) -> str:
    """
    Format Git tree diff counts into a concise summary badge string.
    """
    added = len(changes.get("added", []))
    modified = len(changes.get("modified", []))
    deleted = len(changes.get("deleted", []))
    unchanged = len(changes.get("unchanged", []))
    return f"+{added} added, ~{modified} modified, -{deleted} deleted, ={unchanged} unchanged"


def render_citations_panel(citations: list[dict[str, Any]]) -> None:
    """
    Streamlit renderer for code citations.
    """
    if not citations:
        return

    with st.expander(f"📚 Citations ({len(citations)})", expanded=False):
        for idx, cit in enumerate(citations, 1):
            cit_str = format_citation(cit)
            path = cit.get("path") if isinstance(cit, dict) else None
            start = cit.get("start_line") if isinstance(cit, dict) else None
            end = cit.get("end_line") if isinstance(cit, dict) else None
            repo = cit.get("repo") if isinstance(cit, dict) else None

            badge = f"`{cit_str}`"
            detail = f"File: **{path}**"
            if start is not None and end is not None:
                detail += f" (Lines {start}–{end})"
            elif start is not None:
                detail += f" (Line {start})"
            if repo:
                detail += f" in repository `{repo}`"

            st.markdown(f"- {badge} — {detail}")


def render_source_chunks(sources: list[dict[str, Any]]) -> None:
    """
    Streamlit renderer for retrieved source chunks with syntax highlighting and line numbers.
    """
    if not sources:
        return

    with st.expander(f"🔍 Retrieved Code Chunks ({len(sources)})", expanded=False):
        for idx, src in enumerate(sources, 1):
            path = src.get("path", "unknown")
            start = src.get("start_line")
            end = src.get("end_line")
            repo = src.get("repo")
            lang = src.get("language") or "text"
            sim = src.get("similarity")
            text = src.get("text", "")

            # Header info
            repo_tag = f"[{repo}] " if repo else ""
            line_str = f":{start}-{end}" if start and end else ""
            sim_str = f" (similarity: {sim:.3f})" if sim is not None else ""
            title = f"#{idx} {repo_tag}`{path}{line_str}`{sim_str}"

            st.markdown(f"**{title}**")
            st.code(text, language=lang)
            st.divider()


def render_changes_details(changes: dict[str, Any]) -> None:
    """
    Streamlit renderer for repository diff preview.
    """
    added = changes.get("added", [])
    modified = changes.get("modified", [])
    deleted = changes.get("deleted", [])
    unchanged = changes.get("unchanged", [])

    st.markdown(f"**Summary:** {format_changes_summary(changes)}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Added ({len(added)})**")
        for f in added[:15]:
            st.markdown(f"- 🟢 `{f}`")
        if len(added) > 15:
            st.caption(f"...and {len(added) - 15} more")

    with c2:
        st.markdown(f"**Modified ({len(modified)})**")
        for f in modified[:15]:
            st.markdown(f"- 🟡 `{f}`")
        if len(modified) > 15:
            st.caption(f"...and {len(modified) - 15} more")

    with c3:
        st.markdown(f"**Deleted ({len(deleted)})**")
        for f in deleted[:15]:
            st.markdown(f"- 🔴 `{f}`")
        if len(deleted) > 15:
            st.caption(f"...and {len(deleted) - 15} more")

    st.caption(f"Unchanged files: {len(unchanged)} (will not be re-embedded)")
