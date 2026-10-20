"""
Streamlit Web Application for the AI Codebase Intelligence Platform.
Interactive codebase chat, repository exploration, incremental ingestion, and citation inspection.
"""
from __future__ import annotations

import os
from typing import Any
import streamlit as st

from src.ui.client import ApiClient, ApiClientError, ApiConnectionError
from src.ui.components import (
    format_citation,
    format_metrics,
    format_timestamp,
    get_status_badge,
    format_repo_label,
    render_citations_panel,
    render_source_chunks,
    render_changes_details,
)


# ============================================================================
# Page Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="AI Codebase Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for badges, metrics, and cards
st.markdown(
    """
    <style>
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .stCodeBlock {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .repo-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state() -> None:
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    if "client" not in st.session_state or st.session_state.client.base_url != st.session_state.api_base_url.rstrip("/"):
        st.session_state.client = ApiClient(base_url=st.session_state.api_base_url)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "github_token" not in st.session_state:
        st.session_state.github_token = os.getenv("GITHUB_TOKEN", "")

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


init_session_state()
client: ApiClient = st.session_state.client


# ============================================================================
# Sidebar: System Health & Configuration
# ============================================================================

with st.sidebar:
    st.title("🧠 Codebase Intel")
    st.caption("AI-powered natural language codebase Q&A with grounded citations.")

    st.divider()

    # Health & connection check
    backend_healthy = False
    health_data: dict[str, Any] = {}
    try:
        health_data = client.health()
        backend_healthy = (health_data.get("status") == "ok")
    except Exception:
        backend_healthy = False

    if backend_healthy:
        st.success("🟢 **Backend Connected**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Indexed Repos", health_data.get("indexed_repos_count", 0))
            st.caption(f"Vector: `{health_data.get('vector_store_type', 'Chroma')}`")
        with c2:
            st.metric("Collections", health_data.get("collections_count", 0))
            st.caption(f"LLM: `{health_data.get('llm_provider', 'gemini')}`")
    else:
        st.error("🔴 **Backend Offline**")
        st.warning(
            f"Cannot connect to `{client.base_url}`.\n\n"
            "Start the API server in your terminal:\n"
            "```bash\npython -m src.api.main\n```"
        )

    st.divider()

    # Configuration options
    with st.expander("⚙️ Connection Settings", expanded=False):
        api_input = st.text_input(
            "Backend API URL",
            value=st.session_state.api_base_url,
            help="URL of the running FastAPI server",
        )
        if api_input != st.session_state.api_base_url:
            st.session_state.api_base_url = api_input
            st.session_state.client = ApiClient(base_url=api_input)
            st.rerun()

        token_input = st.text_input(
            "GitHub Token (optional)",
            value=st.session_state.github_token,
            type="password",
            help="Overrides server GITHUB_TOKEN environment variable",
        )
        if token_input != st.session_state.github_token:
            st.session_state.github_token = token_input

    # Quick repo list
    if backend_healthy:
        try:
            repos_list = client.list_repos()
            if repos_list:
                st.subheader("📚 Registered Codebases")
                for r in repos_list:
                    icon, label = get_status_badge(r.get("status"))
                    st.markdown(f"- {icon} **{r.get('owner')}/{r.get('repo')}** ({r.get('chunk_count', 0)} chunks)")
        except Exception:
            pass


# ============================================================================
# Main Navigation Tabs
# ============================================================================

tab_chat, tab_repos, tab_collections, tab_about = st.tabs([
    "💬 Codebase Chat",
    "📦 Repositories & Ingestion",
    "🔍 Vector Collections",
    "ℹ️ Architecture & About",
])


# ============================================================================
# TAB 1: Codebase Chat
# ============================================================================

with tab_chat:
    st.header("💬 Codebase Intelligence Chat")

    # Fetch available repositories
    available_repos: list[dict[str, Any]] = []
    if backend_healthy:
        try:
            available_repos = client.list_repos()
        except Exception as e:
            st.warning(f"Failed to fetch repositories: {e}")

    # Top controls: Repository selection and query parameters
    col_scope, col_params = st.columns([3, 2])

    with col_scope:
        repo_options = ["All Indexed Repositories"] + [f"{r['owner']}/{r['repo']}" for r in available_repos]
        selected_scope = st.selectbox(
            "Target Codebase Scope",
            options=repo_options,
            index=0,
            help="Choose whether to query across all repositories or isolate search to one",
        )

    with col_params:
        with st.popover("⚙️ Retrieval & LLM Settings"):
            top_k = st.slider("Candidate Chunks (Top-K)", min_value=1, max_value=25, value=10)
            max_tokens = st.slider("Context Token Budget", min_value=512, max_value=8192, value=2048, step=256)
            rerank_enabled = st.checkbox("Multi-signal Re-ranking", value=True, help="Rerank based on keyword, path, and definition density")
            merge_enabled = st.checkbox("Merge Contiguous Chunks", value=True, help="Combine adjacent sliding window chunks from the same file")
            temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if not available_repos and backend_healthy:
        st.info("💡 **No repositories indexed yet.** Head to the **Repositories & Ingestion** tab to index a GitHub repository.")

    # Quick prompt buttons
    st.markdown("**Suggested Questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("🔐 How is authentication implemented?", use_container_width=True):
            st.session_state.pending_prompt = "How is authentication implemented in this codebase?"
    with q_col2:
        if st.button("🛣️ What API routes or endpoints exist?", use_container_width=True):
            st.session_state.pending_prompt = "What API routes or endpoints are exposed in this codebase?"
    with q_col3:
        if st.button("🏗️ Explain the high-level architecture", use_container_width=True):
            st.session_state.pending_prompt = "Explain the high-level architecture and key modules of this codebase."

    st.divider()

    # Display Chat History
    for message in st.session_state.messages:
        role = message.get("role", "user")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))

            # Assistant-specific grounded metadata
            if role == "assistant":
                citations = message.get("citations", [])
                if citations:
                    render_citations_panel(citations)

                sources = message.get("sources", [])
                if sources:
                    render_source_chunks(sources)

                metadata = message.get("metadata", {})
                metric_str = format_metrics(metadata)
                if metric_str:
                    st.caption(f"📊 {metric_str}")

    # Process prompt from input or suggestion buttons
    prompt_input = st.chat_input("Ask a question about the code (e.g. 'How does user login work?')...")
    active_prompt = prompt_input or st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if active_prompt:
        if not backend_healthy:
            st.error("Cannot query: backend server is offline. Please start it with `python -m src.api.main`.")
        else:
            # 1. Render and append user prompt
            st.session_state.messages.append({"role": "user", "content": active_prompt})
            with st.chat_message("user"):
                st.markdown(active_prompt)

            # 2. Execute RAG query against API
            with st.chat_message("assistant"):
                with st.spinner("Searching codebase, reranking chunks, and generating cited answer..."):
                    try:
                        target_repo = None
                        target_repos = None
                        if selected_scope != "All Indexed Repositories":
                            target_repo = selected_scope

                        res = client.query(
                            question=active_prompt,
                            repo=target_repo,
                            repos=target_repos,
                            top_k=top_k,
                            max_context_tokens=max_tokens,
                            rerank=rerank_enabled,
                            merge_overlapping=merge_enabled,
                            temperature=temperature,
                        )

                        answer = res.get("formatted_answer") or res.get("answer", "")
                        citations = res.get("citations", [])
                        sources = res.get("sources", [])
                        metadata = res.get("metadata", {})

                        st.markdown(answer)

                        if citations:
                            render_citations_panel(citations)
                        if sources:
                            render_source_chunks(sources)

                        metric_str = format_metrics(metadata)
                        if metric_str:
                            st.caption(f"📊 {metric_str}")

                        # Append assistant response to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "sources": sources,
                            "metadata": metadata,
                        })

                    except ApiClientError as err:
                        error_msg = f"❌ **Query Error:** {err.message}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation History", type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ============================================================================
# TAB 2: Repositories & Ingestion
# ============================================================================

with tab_repos:
    st.header("📦 Repository Management & Ingestion")
    st.caption("Index new GitHub repositories or synchronize existing codebases incrementally.")

    # --- Section: Ingest New Repository ---
    with st.expander("📥 Index a New Repository", expanded=True):
        with st.form("index_repo_form"):
            c_owner, c_repo = st.columns(2)
            with c_owner:
                owner_input = st.text_input("Repository Owner / Organization", value="miguelgrinberg", help="e.g. miguelgrinberg")
            with c_repo:
                repo_input = st.text_input("Repository Name", value="microblog", help="e.g. microblog")

            c_branch, c_opts = st.columns(2)
            with c_branch:
                branch_input = st.text_input("Branch (optional)", value="", placeholder="Leave blank to auto-detect default branch")
            with c_opts:
                st.write("")
                st.write("")
                force_index = st.checkbox("Force full re-index (bypass SHA check)", value=False)
                bg_index = st.checkbox("Run in background", value=False)

            submit_index = st.form_submit_button("🚀 Start Ingestion & Indexing", type="primary")

        if submit_index:
            if not backend_healthy:
                st.error("Backend server is not running.")
            elif not owner_input.strip() or not repo_input.strip():
                st.warning("Please provide both repository owner and name.")
            else:
                with st.spinner(f"Ingesting and indexing `{owner_input.strip()}/{repo_input.strip()}`..."):
                    try:
                        token = st.session_state.github_token.strip() or None
                        branch = branch_input.strip() or None
                        idx_res = client.index_repo(
                            owner=owner_input.strip(),
                            repo=repo_input.strip(),
                            branch=branch,
                            token=token,
                            force=force_index,
                            background=bg_index,
                        )

                        status = idx_res.get("status")
                        commit = (idx_res.get("commit_sha") or "")[:7]
                        chunks = idx_res.get("chunks_stored", 0)
                        files = idx_res.get("files_processed", 0)

                        if bg_index:
                            st.info(f"⏳ **Indexing scheduled in background!** Check status below shortly.")
                        elif status == "unchanged":
                            st.info(f"ℹ️ Repository is already up to date at commit `{commit}` (0 files re-embedded).")
                        else:
                            st.success(
                                f"✅ **Indexed successfully!** Status: `{status}` | Commit: `{commit}` | "
                                f"Files: {files} | Chunks stored: {chunks}"
                            )
                        st.rerun()

                    except ApiClientError as err:
                        st.error(f"❌ **Indexing failed:** {err.message}")

    st.divider()

    # --- Section: Registered Repositories Explorer ---
    st.subheader("📚 Indexed Codebases")

    if not backend_healthy:
        st.warning("Connect to the backend to view indexed repositories.")
    else:
        try:
            repos = client.list_repos()
            if not repos:
                st.info("No repositories indexed yet.")
            else:
                for repo in repos:
                    owner = repo.get("owner", "")
                    rname = repo.get("repo", "")
                    full_name = f"{owner}/{rname}"
                    commit = (repo.get("commit_sha") or "")[:7]
                    status_icon, status_label = get_status_badge(repo.get("status"))

                    with st.expander(f"{status_icon} **{full_name}** — Branch: `{repo.get('branch')}` | Commit: `{commit}` | Chunks: `{repo.get('chunk_count', 0)}`"):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Files Indexed", repo.get("file_count", 0))
                        m2.metric("Total Chunks", repo.get("chunk_count", 0))
                        m3.metric("Status", status_label)
                        m4.metric("Last Indexed", format_timestamp(repo.get("indexed_at")))

                        # Action buttons
                        b1, b2, b3 = st.columns(3)
                        with b1:
                            if st.button("🔍 Check Git Changes", key=f"changes_{full_name}"):
                                with st.spinner("Checking remote GitHub tree for changes..."):
                                    try:
                                        token = st.session_state.github_token.strip() or None
                                        changes = client.get_repo_changes(owner, rname, branch=repo.get("branch"), token=token)
                                        render_changes_details(changes)
                                    except ApiClientError as e:
                                        st.error(f"Diff check failed: {e.message}")

                        with b2:
                            if st.button("🔄 Sync / Re-index", key=f"sync_{full_name}"):
                                with st.spinner(f"Synchronizing `{full_name}` incrementally..."):
                                    try:
                                        token = st.session_state.github_token.strip() or None
                                        sync_res = client.index_repo(owner, rname, branch=repo.get("branch"), token=token)
                                        st.success(f"Sync complete: {sync_res.get('status')} ({sync_res.get('chunks_stored', 0)} chunks stored)")
                                        st.rerun()
                                    except ApiClientError as e:
                                        st.error(f"Sync failed: {e.message}")

                        with b3:
                            if st.button("🗑️ Delete Repository", key=f"del_{full_name}", type="secondary"):
                                try:
                                    client.delete_repo(owner, rname)
                                    st.success(f"Deleted `{full_name}` and purged vectors.")
                                    st.rerun()
                                except ApiClientError as e:
                                    st.error(f"Deletion failed: {e.message}")

        except Exception as e:
            st.error(f"Error loading repositories: {e}")


# ============================================================================
# TAB 3: Vector Collections
# ============================================================================

with tab_collections:
    st.header("🔍 Vector Store Collections")
    st.caption("Inspect underlying vector store collections, chunk counts, and sample indexed snippets.")

    if not backend_healthy:
        st.warning("Connect to the backend to view vector store collections.")
    else:
        try:
            collections = client.list_collections()
            if not collections:
                st.info("No vector collections found in storage.")
            else:
                col_names = [c["name"] for c in collections]
                selected_col_name = st.selectbox("Select Vector Collection to Inspect", options=col_names)

                if selected_col_name:
                    col_detail = client.get_collection(selected_col_name)
                    c_info1, c_info2 = st.columns(2)
                    c_info1.metric("Collection Name", col_detail.get("name", ""))
                    c_info2.metric("Total Stored Chunks", col_detail.get("chunk_count", 0))

                    sample_chunks = col_detail.get("sample_chunks", [])
                    if sample_chunks:
                        st.subheader(f"Sample Chunks ({len(sample_chunks)})")
                        for idx, chunk in enumerate(sample_chunks, 1):
                            meta = chunk.get("metadata", {})
                            path = meta.get("path") or chunk.get("id", "chunk")
                            start = meta.get("start_line")
                            end = meta.get("end_line")
                            lang = meta.get("language") or "text"
                            line_info = f":{start}-{end}" if start and end else ""

                            with st.expander(f"Chunk #{idx} — `{path}{line_info}`"):
                                st.code(chunk.get("text", ""), language=lang)
                    else:
                        st.caption("No sample chunks available.")

                    st.divider()
                    if st.button("🗑️ Purge Collection", key=f"purge_{selected_col_name}", type="secondary"):
                        try:
                            client.delete_collection(selected_col_name)
                            st.success(f"Purged collection `{selected_col_name}`.")
                            st.rerun()
                        except ApiClientError as e:
                            st.error(f"Purge failed: {e.message}")

        except Exception as e:
            st.error(f"Error loading vector collections: {e}")


# ============================================================================
# TAB 4: Architecture & About
# ============================================================================

with tab_about:
    st.header("ℹ️ System Architecture & Design")

    st.markdown(
        """
        The **AI Codebase Intelligence Platform** enables grounded natural-language exploration of complex software repositories.
        Unlike naive LLM ingestion which suffers from token limits, excessive latency, and hallucinated answers, this platform
        uses a specialized multi-stage RAG architecture designed specifically for code search.
        """
    )

    st.subheader("🛠️ The Pipeline")
    st.markdown(
        """
        1. **Ingestion & Filtering (`src/ingestion/`):**
           - Pulls remote file trees via the GitHub API without cloning git history.
           - Applies an intelligent blocklist filter removing binary files, build artifacts, test caches, lockfiles, and git metadata.
        2. **Language-Agnostic Chunking (`src/ingestion/chunker.py`):**
           - Overlapping sliding-window chunking (`CHUNK_SIZE=400`, `OVERLAP=50`).
           - Boundary snapping to the nearest blank line (`BOUNDARY_SEARCH_WINDOW=5`) preserves function and block boundaries without brittle AST parsers.
        3. **Code Embedding (`src/embedding/embedder.py`):**
           - `st-codesearch-distilroberta-base` encodes semantic code structures into 768-dimensional vector spaces.
        4. **Vector DB Storage (`src/storage/`):**
           - Chroma vector store with persistent local storage and cosine similarity distance metric.
        5. **Incremental Git Sync (`src/registry/`):**
           - Merkle tree blob SHA diffing detects added, modified, deleted, and unchanged files.
           - Unchanged files are never re-fetched or re-embedded.
        6. **Multi-Signal Re-Ranking (`src/rag/reranker.py`):**
           - Combines dense vector similarity with lexical keyword matching, symbol definition recognition (`def`, `class`), and path relevance.
        7. **Context Token Budgeting (`src/rag/context_assembler.py`):**
           - Strictly enforces token limits with line-preserving truncation and "lost in the middle" attention ordering.
        8. **Grounded Generation & Citations (`src/rag/citations.py`, `src/llm/`):**
           - Citations cite exact file paths and line ranges (`path/to/file.py:10-25`).
        """
    )

    st.divider()
    st.subheader("🔗 API & Integration")
    st.markdown(
        f"""
        - **FastAPI Documentation:** [Swagger UI]({client.base_url}/docs) | [ReDoc]({client.base_url}/redoc)
        - **API Health Endpoint:** `{client.base_url}/health`
        - **CLI Runner:** `python -m src.ui.main --port 8501`
        """
    )
