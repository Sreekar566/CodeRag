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

    duration_ms = metadata.get("duration_ms")
    elapsed = metadata.get("elapsed_seconds")
    if duration_ms is not None:
        parts.append(f"Time: {duration_ms:.0f}ms")
    elif elapsed is not None:
        parts.append(f"Time: {elapsed:.2f}s")

    cost = metadata.get("estimated_cost_usd")
    if cost is not None:
        parts.append(f"Cost: ${cost:.6f}" if cost > 0 else "Cost: $0.00")

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


def format_latency_pill(
    duration_ms: float | None,
    stage_durations: dict[str, float] | None = None,
) -> str:
    """Format total latency and stage breakdown into a readable pill."""
    if duration_ms is None:
        return ""
    if not stage_durations:
        return f"⏱️ {duration_ms:.1f} ms"

    stages = []
    if "retrieval" in stage_durations:
        stages.append(f"Retrieval: {stage_durations['retrieval']:.1f}ms")
    if "generation" in stage_durations:
        stages.append(f"Generation: {stage_durations['generation']:.1f}ms")

    stage_str = f" ({' | '.join(stages)})" if stages else ""
    return f"⏱️ {duration_ms:.1f} ms{stage_str}"


def format_cost_badge(cost_usd: float | None) -> str:
    """Format query cost into a badge."""
    if cost_usd is None or cost_usd == 0:
        return "💰 $0.00"
    return f"💰 ${cost_usd:.6f}"


def render_audit_log(audit: dict[str, Any]) -> None:
    """Render repository ingestion transparency audit log in Streamlit."""
    total_scanned = audit.get("total_scanned", 0)
    total_kept = audit.get("total_kept", 0)
    total_filtered = audit.get("total_filtered", 0)
    reason_counts = audit.get("reason_counts", {})
    filtered_files = audit.get("filtered_files", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Scanned", f"{total_scanned:,}")
    c2.metric("Files Kept (Indexed)", f"{total_kept:,}")
    c3.metric("Files Filtered Out", f"{total_filtered:,}")

    if reason_counts:
        st.markdown("**Filter Reason Breakdown:**")
        cols = st.columns(len(reason_counts))
        for col, (reason, count) in zip(cols, sorted(reason_counts.items(), key=lambda x: -x[1])):
            friendly_name = reason.replace("_", " ").title()
            col.metric(friendly_name, f"{count:,}")

    if filtered_files:
        with st.expander(f"🔍 Inspect Filtered Files ({len(filtered_files)} files)", expanded=False):
            search = st.text_input("Filter paths...", key=f"audit_search_{audit.get('repo_id', '')}")
            displayed = filtered_files
            if search:
                displayed = [f for f in filtered_files if search.lower() in f.get("path", "").lower()]

            table_rows = []
            for f in displayed[:100]:
                size_str = f"{f.get('size_bytes', 0):,} B" if f.get("size_bytes") is not None else "N/A"
                table_rows.append({
                    "Path": f.get("path"),
                    "Reason": f.get("reason"),
                    "Rule Matched": f.get("rule_match"),
                    "Size": size_str,
                })
            if table_rows:
                st.dataframe(table_rows, use_container_width=True)
                if len(displayed) > 100:
                    st.caption(f"Showing first 100 of {len(displayed)} matching files.")
            else:
                st.info("No matching files found.")


def render_metrics_dashboard(metrics: dict[str, Any], queries: list[dict[str, Any]]) -> None:
    """Render overall observability & cost monitoring telemetry dashboard."""
    m1, m2, m3, m4 = st.columns(4)
    total_q = metrics.get("total_queries", 0)
    total_tokens = metrics.get("total_tokens", 0)
    total_cost = metrics.get("total_cost_usd", 0.0)
    avg_latency = metrics.get("avg_total_latency_ms", 0.0)

    m1.metric("Total Queries", f"{total_q:,}")
    m2.metric("Tokens Consumed", f"{total_tokens:,}")
    m3.metric("Estimated Cost", f"${total_cost:.5f}")
    m4.metric("Avg Latency", f"{avg_latency:.1f} ms", f"p95: {metrics.get('p95_latency_ms', 0):.1f} ms")

    st.divider()

    # Latency Stage Breakdown
    stage_latencies = metrics.get("avg_stage_latencies_ms", {})
    if stage_latencies and any(v > 0 for v in stage_latencies.values()):
        st.subheader("⚡ Latency by Pipeline Stage")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Semantic Retrieval", f"{stage_latencies.get('retrieval', 0):.1f} ms")
        sc2.metric("Context & Rerank", f"{stage_latencies.get('assembly_rerank', 0):.1f} ms")
        sc3.metric("LLM Generation", f"{stage_latencies.get('generation', 0):.1f} ms")
        sc4.metric("Citation Extraction", f"{stage_latencies.get('citations', 0):.1f} ms")

    # Recent Queries Log
    st.subheader("📋 Recent Query Activity")
    if queries:
        query_rows = []
        for q in queries:
            query_rows.append({
                "Timestamp": format_timestamp(q.get("timestamp")),
                "Question": q.get("question", "")[:60] + "..." if len(q.get("question", "")) > 60 else q.get("question", ""),
                "Model": q.get("model", ""),
                "Duration": f"{q.get('total_duration_ms', 0):.1f} ms",
                "Tokens": q.get("total_tokens") or "N/A",
                "Cost ($)": f"${q.get('cost_usd', 0.0):.6f}",
                "Citations": q.get("citations_count", 0),
            })
        st.dataframe(query_rows, use_container_width=True)
    else:
        st.info("No queries executed yet. Ask questions in the **Codebase Chat** tab to populate telemetry.")

