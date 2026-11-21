"""
Comprehensive unit tests for the Streamlit UI components, API client, and CLI runner.
All tests run offline without network access or running backend servers.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.ui.client import ApiClient, ApiClientError, ApiConnectionError, ApiTimeoutError
from src.ui.components import (
    format_citation,
    format_metrics,
    format_timestamp,
    get_status_badge,
    format_repo_label,
    format_changes_summary,
    render_changes_details,
    render_citations_panel,
    render_source_chunks,
    render_audit_log,
    render_metrics_dashboard,
    format_latency_pill,
    format_cost_badge,
    _extract_changes_list,
)
from src.ui.main import parse_args, get_app_path, main as ui_main


# ============================================================================
# ApiClient Tests
# ============================================================================

def test_api_client_init_and_url_normalization(monkeypatch):
    """Test URL normalization and default fallback."""
    monkeypatch.delenv("API_BASE_URL", raising=False)
    client = ApiClient()
    assert client.base_url == "http://localhost:8000"

    client2 = ApiClient(base_url="http://custom-api:9000/")
    assert client2.base_url == "http://custom-api:9000"

    monkeypatch.setenv("API_BASE_URL", "http://env-api:8080/")
    client3 = ApiClient()
    assert client3.base_url == "http://env-api:8080"


def test_api_client_health_success():
    """Test health check parsing on valid response."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "status": "ok",
        "vector_store_type": "ChromaVectorStore",
        "llm_provider": "gemini",
        "indexed_repos_count": 2,
        "collections_count": 2,
    }

    with patch.object(client.session, "request", return_value=fake_resp) as mock_req:
        res = client.health()
        assert res["status"] == "ok"
        assert res["indexed_repos_count"] == 2
        mock_req.assert_called_once_with(
            method="GET",
            url="http://fake-api:8000/health",
            params=None,
            json=None,
            timeout=5.0,
        )


def test_api_client_is_connected():
    """Test is_connected helper for success and failure."""
    client = ApiClient("http://fake-api:8000")

    # Success case
    with patch.object(client, "health", return_value={"status": "ok"}):
        assert client.is_connected() is True

    # Error case
    with patch.object(client, "health", side_effect=ApiConnectionError("Offline")):
        assert client.is_connected() is False


def test_api_client_connection_error():
    """Test connection failure raises ApiConnectionError."""
    client = ApiClient("http://unreachable:8000")
    with patch.object(
        client.session,
        "request",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        with pytest.raises(ApiConnectionError) as exc_info:
            client.health()
        assert "Failed to connect to backend server" in str(exc_info.value)


def test_api_client_timeout_error():
    """Test request timeout raises ApiConnectionError / ApiTimeoutError."""
    client = ApiClient("http://fake-api:8000")
    with patch.object(
        client.session,
        "request",
        side_effect=requests.exceptions.Timeout("Request timed out"),
    ):
        with pytest.raises(ApiConnectionError) as exc_info:
            client.health()
        assert "Failed to connect to backend server" in str(exc_info.value)
        assert isinstance(exc_info.value, ApiTimeoutError)


def test_api_client_http_error_json():
    """Test HTTP 404/400 errors extract JSON detail message."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 404
    fake_resp.json.return_value = {"detail": "Repository not found"}

    with patch.object(client.session, "request", return_value=fake_resp):
        with pytest.raises(ApiClientError) as exc_info:
            client.get_repo("owner", "nonexistent")
        assert "Repository not found" in str(exc_info.value)
        assert exc_info.value.status_code == 404


def test_api_client_list_repos():
    """Test list_repos retrieves array from response."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "repos": [
            {"owner": "miguelgrinberg", "repo": "microblog", "chunk_count": 50},
        ],
        "total": 1,
    }

    with patch.object(client.session, "request", return_value=fake_resp):
        repos = client.list_repos()
        assert len(repos) == 1
        assert repos[0]["repo"] == "microblog"


def test_api_client_get_and_delete_repo():
    """Test get_repo and delete_repo API calls."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"owner": "test", "repo": "app", "chunk_count": 10}

    with patch.object(client.session, "request", return_value=fake_resp) as mock_req:
        res = client.get_repo("test", "app")
        assert res["repo"] == "app"
        assert mock_req.call_args[1]["url"] == "http://fake-api:8000/repos/test/app"

    fake_del_resp = MagicMock()
    fake_del_resp.status_code = 200
    fake_del_resp.json.return_value = {"status": "deleted", "owner": "test", "repo": "app"}

    with patch.object(client.session, "request", return_value=fake_del_resp) as mock_del:
        del_res = client.delete_repo("test", "app")
        assert del_res["status"] == "deleted"
        assert mock_del.call_args[1]["url"] == "http://fake-api:8000/repos/test/app"


def test_api_client_get_repo_changes():
    """Test get_repo_changes passing branch and token query parameters."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "owner": "test",
        "repo": "app",
        "added": ["new.py"],
        "modified": [],
        "deleted": [],
        "unchanged": ["main.py"],
    }

    with patch.object(client.session, "request", return_value=fake_resp) as mock_req:
        res = client.get_repo_changes("test", "app", branch="feature", token="ghp_sec")
        assert res["added"] == ["new.py"]
        assert mock_req.call_args[1]["params"] == {"branch": "feature", "token": "ghp_sec"}


def test_api_client_index_repo():
    """Test index_repo constructs correct JSON payload."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "owner": "miguelgrinberg",
        "repo": "microblog",
        "collection": "repo_miguelgrinberg_microblog",
        "status": "created",
        "files_processed": 54,
        "chunks_stored": 120,
    }

    with patch.object(client.session, "request", return_value=fake_resp) as mock_req:
        res = client.index_repo(
            owner="miguelgrinberg",
            repo="microblog",
            branch="main",
            token="ghp_token",
            force=True,
            background=False,
        )
        assert res["status"] == "created"
        payload = mock_req.call_args[1]["json"]
        assert payload["owner"] == "miguelgrinberg"
        assert payload["repo"] == "microblog"
        assert payload["branch"] == "main"
        assert payload["token"] == "ghp_token"
        assert payload["force"] is True
        assert payload["background"] is False


def test_api_client_index_repo_sync_param():
    """Test index_repo correctly handles sync parameter for backward compatibility."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"status": "created"}

    with patch.object(client.session, "request", return_value=fake_resp) as mock_req:
        client.index_repo(owner="foo", repo="bar", sync=True)
        assert mock_req.call_args[1]["json"]["background"] is False

    with patch.object(client.session, "request", return_value=fake_resp) as mock_req:
        client.index_repo(owner="foo", repo="bar", sync=False)
        assert mock_req.call_args[1]["json"]["background"] is True


def test_api_client_query():
    """Test query sends parameters and returns response structure."""
    client = ApiClient("http://fake-api:8000")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "query": "How does login work?",
        "answer": "Login is in app/auth.py",
        "formatted_answer": "Login is in app/auth.py",
        "citations": [{"citation_str": "app/auth.py:10-25", "path": "app/auth.py"}],
        "sources": [{"id": "chunk_1", "text": "def login():", "path": "app/auth.py"}],
        "model": "gemini-3.8-flash",
        "metadata": {"context_tokens": 150, "retrieved_count": 5, "selected_count": 2},
    }

    with patch.object(client.session, "request", return_value=fake_resp) as mock_req:
        res = client.query(
            question="How does login work?",
            repo="miguelgrinberg/microblog",
            top_k=8,
            max_context_tokens=1024,
            rerank=True,
            temperature=0.3,
        )
        assert res["query"] == "How does login work?"
        assert len(res["citations"]) == 1
        assert len(res["sources"]) == 1
        payload = mock_req.call_args[1]["json"]
        assert payload["repo"] == "miguelgrinberg/microblog"
        assert payload["top_k"] == 8
        assert payload["max_context_tokens"] == 1024
        assert payload["temperature"] == 0.3


def test_api_client_collections():
    """Test list_collections, get_collection, delete_collection."""
    client = ApiClient("http://fake-api:8000")

    fake_list = MagicMock()
    fake_list.status_code = 200
    fake_list.json.return_value = {
        "collections": [{"name": "repo_microblog", "chunk_count": 42}],
        "total": 1,
    }
    with patch.object(client.session, "request", return_value=fake_list):
        cols = client.list_collections()
        assert len(cols) == 1
        assert cols[0]["name"] == "repo_microblog"

    fake_detail = MagicMock()
    fake_detail.status_code = 200
    fake_detail.json.return_value = {"name": "repo_microblog", "chunk_count": 42, "sample_chunks": []}
    with patch.object(client.session, "request", return_value=fake_detail):
        col = client.get_collection("repo_microblog")
        assert col["chunk_count"] == 42

    fake_del = MagicMock()
    fake_del.status_code = 200
    fake_del.json.return_value = {"status": "deleted", "name": "repo_microblog"}
    with patch.object(client.session, "request", return_value=fake_del):
        del_res = client.delete_collection("repo_microblog")
        assert del_res["status"] == "deleted"


# ============================================================================
# Component and Formatting Helper Tests
# ============================================================================

def test_format_citation():
    """Test format_citation handling dicts and strings."""
    # From citation_str
    assert format_citation({"citation_str": "app/auth.py:10-25"}) == "app/auth.py:10-25"

    # Computed from span
    c1 = {"path": "src/main.py", "start_line": 5, "end_line": 20}
    assert format_citation(c1) == "src/main.py:5-20"

    # With repo
    c2 = {"repo": "owner/repo", "path": "src/main.py", "start_line": 5, "end_line": 20}
    assert format_citation(c2) == "owner/repo:src/main.py:5-20"

    # Start line only
    c3 = {"path": "src/main.py", "start_line": 15}
    assert format_citation(c3) == "src/main.py:15"

    # Path only
    c4 = {"path": "README.md"}
    assert format_citation(c4) == "README.md"

    # String pass-through
    assert format_citation("direct_string:1-10") == "direct_string:1-10"

    # Non-dict and non-string fallback
    assert format_citation(12345) == "12345"


def test_format_metrics():
    """Test format_metrics output formatting."""
    assert format_metrics(None) == ""
    assert format_metrics({}) == ""

    meta = {
        "context_tokens": 420,
        "max_context_tokens": 2048,
        "retrieved_count": 10,
        "selected_count": 4,
        "reranked": True,
        "elapsed_seconds": 1.25,
    }
    s = format_metrics(meta)
    assert "Tokens: 420/2048" in s
    assert "Chunks: 4/10" in s
    assert "Re-ranked: Yes" in s
    assert "Time: 1.25s" in s

    # Alternate formatting branches: duration_ms, cost formatting, no max_tokens, no selected_count
    meta_alt = {
        "context_tokens": 150,
        "retrieved_count": 5,
        "duration_ms": 250.4,
        "estimated_cost_usd": 0.000125,
    }
    s_alt = format_metrics(meta_alt)
    assert "Tokens: 150" in s_alt
    assert "Chunks: 5" in s_alt
    assert "Time: 250ms" in s_alt
    assert "Cost: $0.000125" in s_alt

    # Zero cost
    assert "Cost: $0.00" in format_metrics({"estimated_cost_usd": 0.0})


def test_format_timestamp():
    """Test format_timestamp date cleaner."""
    assert format_timestamp(None) == "N/A"
    assert format_timestamp("") == "N/A"
    assert format_timestamp("2026-10-21T00:00:00.123456") == "2026-10-21 00:00:00 UTC"
    # Exception fallback when value cannot be string-replaced like ISO timestamp
    assert format_timestamp(12345) == "12345"


def test_get_status_badge():
    """Test status badge icon and label mapping."""
    icon, label = get_status_badge("ready")
    assert icon == "🟢"
    assert label == "Ready"

    icon, label = get_status_badge("indexing")
    assert icon == "🟡"
    assert label == "Indexing"

    icon, label = get_status_badge("failed")
    assert icon == "🔴"
    assert label == "Failed"

    icon, label = get_status_badge("unchanged")
    assert icon == "⚪"
    assert label == "Unchanged"

    icon, label = get_status_badge(None)
    assert icon == "⚪"
    assert label == "Unknown"

    # Unknown / other status
    icon, label = get_status_badge("archived")
    assert icon == "ℹ️"
    assert label == "Archived"


def test_format_repo_label():
    """Test format_repo_label helper."""
    repo = {
        "owner": "miguelgrinberg",
        "repo": "microblog",
        "file_count": 54,
        "chunk_count": 110,
        "commit_sha": "a1b2c3d4e5f6",
    }
    label = format_repo_label(repo)
    assert "miguelgrinberg/microblog" in label
    assert "54 files" in label
    assert "110 chunks" in label
    assert "@ a1b2c3d" in label


def test_format_changes_summary():
    """Test format_changes_summary with short keys."""
    changes = {
        "added": ["a.py", "b.py"],
        "modified": ["c.py"],
        "deleted": [],
        "unchanged": ["d.py", "e.py"],
    }
    summary = format_changes_summary(changes)
    assert summary == "+2 added, ~1 modified, -0 deleted, =2 unchanged"


def test_format_changes_summary_with_api_schema():
    """Test format_changes_summary with FastAPI ChangeSetResponse field names."""
    changes = {
        "added_files": ["src/new_feature.py", "src/auth.py"],
        "modified_files": ["src/routes.py"],
        "deleted_files": ["src/legacy.py"],
        "unchanged_files": ["src/main.py", "README.md", "pyproject.toml"],
        "has_changes": True,
        "total_files": 7,
    }
    summary = format_changes_summary(changes)
    assert summary == "+2 added, ~1 modified, -1 deleted, =3 unchanged"


def test_extract_changes_list_fallback():
    """Verify _extract_changes_list returns empty list when keys do not exist or are not lists."""
    assert _extract_changes_list({}, "added_files", "added") == []
    assert _extract_changes_list({"added_files": "not-a-list"}, "added_files") == []


def test_format_latency_pill():
    """Verify format_latency_pill formats total duration and stage breakdown."""
    assert format_latency_pill(None) == ""
    assert format_latency_pill(120.4) == "⏱️ 120.4 ms"
    pill = format_latency_pill(250.0, {"retrieval": 50.0, "generation": 180.0})
    assert pill == "⏱️ 250.0 ms (Retrieval: 50.0ms | Generation: 180.0ms)"
    # Stage dict with missing/other keys
    pill_partial = format_latency_pill(100.0, {"citations": 10.0})
    assert pill_partial == "⏱️ 100.0 ms"


def test_format_cost_badge():
    """Verify format_cost_badge formats cost or defaults to zero."""
    assert format_cost_badge(None) == "💰 $0.00"
    assert format_cost_badge(0.0) == "💰 $0.00"
    assert format_cost_badge(0.000125) == "💰 $0.000125"


def test_render_changes_details_with_both_key_formats():
    """Test render_changes_details formats columns correctly with both key formats."""
    # 1. API schema format
    api_changes = {
        "added_files": ["app/new.py"],
        "modified_files": ["app/mod.py"],
        "deleted_files": ["app/del.py"],
        "unchanged_files": ["app/keep.py"],
    }
    with patch("src.ui.components.st") as mock_st:
        c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = (c1, c2, c3)
        render_changes_details(api_changes)

        mock_st.markdown.assert_any_call("**Summary:** +1 added, ~1 modified, -1 deleted, =1 unchanged")
        mock_st.markdown.assert_any_call("**Added (1)**")
        mock_st.markdown.assert_any_call("- 🟢 `app/new.py`")
        mock_st.markdown.assert_any_call("**Modified (1)**")
        mock_st.markdown.assert_any_call("- 🟡 `app/mod.py`")
        mock_st.markdown.assert_any_call("**Deleted (1)**")
        mock_st.markdown.assert_any_call("- 🔴 `app/del.py`")
        mock_st.caption.assert_called_with("Unchanged files: 1 (will not be re-embedded)")

    # 2. Legacy short key format
    short_changes = {
        "added": ["app/new.py"],
        "modified": ["app/mod.py"],
        "deleted": [],
        "unchanged": [],
    }
    with patch("src.ui.components.st") as mock_st:
        c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = (c1, c2, c3)
        render_changes_details(short_changes)

        mock_st.markdown.assert_any_call("**Summary:** +1 added, ~1 modified, -0 deleted, =0 unchanged")
        mock_st.markdown.assert_any_call("**Added (1)**")
        mock_st.markdown.assert_any_call("**Modified (1)**")
        mock_st.markdown.assert_any_call("**Deleted (0)**")


def test_changeset_response_model_dual_keys():
    """Verify ChangeSetResponse supports both *_files and short keys."""
    from src.api.models import ChangeSetResponse

    resp = ChangeSetResponse(
        owner="testowner",
        repo="testrepo",
        branch="main",
        added_files=["a.py"],
        modified_files=["b.py"],
        deleted_files=["c.py"],
        unchanged_files=["d.py"],
        has_changes=True,
        total_files=4,
    )

    data = resp.model_dump()
    assert data["added_files"] == ["a.py"]
    assert data["added"] == ["a.py"]
    assert data["modified_files"] == ["b.py"]
    assert data["modified"] == ["b.py"]
    assert data["deleted_files"] == ["c.py"]
    assert data["deleted"] == ["c.py"]
    assert data["unchanged_files"] == ["d.py"]
    assert data["unchanged"] == ["d.py"]

    # Remap from short keys
    from_short = ChangeSetResponse(
        owner="testowner",
        repo="testrepo",
        branch="main",
        added=["a.py"],
        modified=["b.py"],
    )
    assert from_short.added_files == ["a.py"]
    assert from_short.added == ["a.py"]
    assert from_short.modified_files == ["b.py"]
    assert from_short.modified == ["b.py"]


# ============================================================================
# Streamlit Rendering Component Tests
# ============================================================================

def test_render_citations_panel_empty():
    """Verify render_citations_panel returns early when citations is empty."""
    with patch("src.ui.components.st") as mock_st:
        render_citations_panel([])
        mock_st.expander.assert_not_called()


def test_render_citations_panel_populated():
    """Verify render_citations_panel formats citations with various line and repo metadata."""
    citations = [
        {"path": "app/auth.py", "start_line": 10, "end_line": 25, "repo": "octocat/hello"},
        {"path": "app/models.py", "start_line": 42},
        {"path": "README.md"},
        "custom_doc.py:1-5",
    ]
    with patch("src.ui.components.st") as mock_st:
        mock_exp = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = mock_exp

        render_citations_panel(citations)

        mock_st.expander.assert_called_once_with("📚 Citations (4)", expanded=False)
        mock_st.markdown.assert_any_call(
            "- `octocat/hello:app/auth.py:10-25` — File: **app/auth.py** (Lines 10–25) in repository `octocat/hello`"
        )
        mock_st.markdown.assert_any_call("- `app/models.py:42` — File: **app/models.py** (Line 42)")
        mock_st.markdown.assert_any_call("- `README.md` — File: **README.md**")
        mock_st.markdown.assert_any_call("- `custom_doc.py:1-5` — File: **None**")


def test_render_source_chunks_empty():
    """Verify render_source_chunks returns early when sources is empty."""
    with patch("src.ui.components.st") as mock_st:
        render_source_chunks([])
        mock_st.expander.assert_not_called()


def test_render_source_chunks_populated():
    """Verify render_source_chunks displays header, code snippet, and divider."""
    sources = [
        {
            "path": "src/utils.py",
            "start_line": 1,
            "end_line": 10,
            "repo": "octocat/hello",
            "language": "python",
            "similarity": 0.8912,
            "text": "def hello():\n    return 'world'",
        },
        {
            "path": "notes.txt",
            "text": "plain text note",
        },
    ]
    with patch("src.ui.components.st") as mock_st:
        mock_exp = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = mock_exp

        render_source_chunks(sources)

        mock_st.expander.assert_called_once_with("🔍 Retrieved Code Chunks (2)", expanded=False)
        mock_st.markdown.assert_any_call("**#1 [octocat/hello] `src/utils.py:1-10` (similarity: 0.891)**")
        mock_st.code.assert_any_call("def hello():\n    return 'world'", language="python")
        mock_st.markdown.assert_any_call("**#2 `notes.txt`**")
        mock_st.code.assert_any_call("plain text note", language="text")
        assert mock_st.divider.call_count == 2


def test_render_changes_details_overflow_truncation():
    """Verify render_changes_details renders overflow captions when lists exceed 15 files."""
    changes = {
        "added": [f"add_{i}.py" for i in range(18)],
        "modified": [f"mod_{i}.py" for i in range(17)],
        "deleted": [f"del_{i}.py" for i in range(16)],
        "unchanged": [],
    }
    with patch("src.ui.components.st") as mock_st:
        c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = (c1, c2, c3)

        render_changes_details(changes)

        mock_st.caption.assert_any_call("...and 3 more")
        mock_st.caption.assert_any_call("...and 2 more")
        mock_st.caption.assert_any_call("...and 1 more")


def test_render_audit_log_metrics_and_breakdown():
    """Verify render_audit_log renders overall metrics and reason columns."""
    audit = {
        "repo_id": "testorg/testrepo",
        "total_scanned": 1200,
        "total_kept": 800,
        "total_filtered": 400,
        "reason_counts": {"test_file": 250, "noise_directory": 150},
        "filtered_files": [],
    }
    with patch("src.ui.components.st") as mock_st:
        cols_3 = [MagicMock(), MagicMock(), MagicMock()]
        cols_breakdown = [MagicMock(), MagicMock()]
        mock_st.columns.side_effect = lambda n: cols_breakdown if n == 2 else cols_3

        render_audit_log(audit)

        cols_3[0].metric.assert_called_once_with("Total Scanned", "1,200")
        cols_3[1].metric.assert_called_once_with("Files Kept (Indexed)", "800")
        cols_3[2].metric.assert_called_once_with("Files Filtered Out", "400")

        mock_st.markdown.assert_any_call("**Filter Reason Breakdown:**")
        cols_breakdown[0].metric.assert_called_once_with("Test File", "250")
        cols_breakdown[1].metric.assert_called_once_with("Noise Directory", "150")
        mock_st.expander.assert_not_called()


def test_render_audit_log_filtered_files_search_and_table():
    """Verify search filter and dataframe display in render_audit_log."""
    filtered = [
        {"path": "tests/test_api.py", "reason": "test_file", "rule_match": "tests/*", "size_bytes": 2048},
        {"path": ".github/workflows/ci.yml", "reason": "ci_config", "rule_match": ".github/*", "size_bytes": None},
    ]
    audit = {
        "repo_id": "testorg/testrepo",
        "total_scanned": 10,
        "total_kept": 8,
        "total_filtered": 2,
        "reason_counts": {},
        "filtered_files": filtered,
    }

    # Case 1: Empty search query displays all rows
    with patch("src.ui.components.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_exp = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = mock_exp
        mock_st.text_input.return_value = ""

        render_audit_log(audit)

        assert mock_st.dataframe.called
        rows = mock_st.dataframe.call_args[0][0]
        assert len(rows) == 2
        assert rows[0]["Path"] == "tests/test_api.py"
        assert rows[0]["Size"] == "2,048 B"
        assert rows[1]["Size"] == "N/A"

    # Case 2: Matching search query filters dataframe rows
    with patch("src.ui.components.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_exp = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = mock_exp
        mock_st.text_input.return_value = "workflows"

        render_audit_log(audit)

        assert mock_st.dataframe.called
        rows = mock_st.dataframe.call_args[0][0]
        assert len(rows) == 1
        assert rows[0]["Path"] == ".github/workflows/ci.yml"

    # Case 3: Non-matching search query shows info message
    with patch("src.ui.components.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_exp = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = mock_exp
        mock_st.text_input.return_value = "nonexistent_subpath"

        render_audit_log(audit)

        mock_st.info.assert_called_once_with("No matching files found.")


def test_render_audit_log_filtered_files_overflow_caption():
    """Verify render_audit_log captions when filtered files exceed 100 entries."""
    overflow_files = [
        {"path": f"noise_{i}.py", "reason": "vendor", "rule_match": "noise/*", "size_bytes": 100}
        for i in range(105)
    ]
    audit = {
        "repo_id": "testorg/testrepo",
        "total_scanned": 150,
        "total_kept": 45,
        "total_filtered": 105,
        "reason_counts": {},
        "filtered_files": overflow_files,
    }
    with patch("src.ui.components.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_exp = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = mock_exp
        mock_st.text_input.return_value = ""

        render_audit_log(audit)

        assert mock_st.dataframe.called
        rows = mock_st.dataframe.call_args[0][0]
        assert len(rows) == 100
        mock_st.caption.assert_called_once_with("Showing first 100 of 105 matching files.")


def test_render_metrics_dashboard_populated():
    """Verify render_metrics_dashboard displays high-level KPIs, stage latencies, and queries log."""
    metrics = {
        "total_queries": 42,
        "total_tokens": 12500,
        "total_cost_usd": 0.0125,
        "avg_total_latency_ms": 150.4,
        "p95_latency_ms": 320.8,
        "avg_stage_latencies_ms": {
            "retrieval": 40.0,
            "assembly_rerank": 15.0,
            "generation": 90.0,
            "citations": 5.4,
        },
    }
    long_question = "How is user authentication implemented in the codebase when the question exceeds sixty characters in total length?"
    short_question = "Where is config defined?"
    queries = [
        {
            "timestamp": "2026-11-16T12:00:00.000000",
            "question": long_question,
            "model": "fake-llm",
            "total_duration_ms": 140.2,
            "total_tokens": 350,
            "cost_usd": 0.00035,
            "citations_count": 2,
        },
        {
            "timestamp": "2026-11-16T12:05:00.000000",
            "question": short_question,
            "model": "fake-llm",
            "total_duration_ms": 95.0,
            "total_tokens": None,
            "cost_usd": 0.0,
            "citations_count": 1,
        },
    ]

    with patch("src.ui.components.st") as mock_st:
        m1, m2, m3, m4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        sc1, sc2, sc3, sc4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.side_effect = [[m1, m2, m3, m4], [sc1, sc2, sc3, sc4]]

        render_metrics_dashboard(metrics, queries)

        m1.metric.assert_called_once_with("Total Queries", "42")
        m2.metric.assert_called_once_with("Tokens Consumed", "12,500")
        m3.metric.assert_called_once_with("Estimated Cost", "$0.01250")
        m4.metric.assert_called_once_with("Avg Latency", "150.4 ms", "p95: 320.8 ms")

        mock_st.subheader.assert_any_call("⚡ Latency by Pipeline Stage")
        sc1.metric.assert_called_once_with("Semantic Retrieval", "40.0 ms")
        sc2.metric.assert_called_once_with("Context & Rerank", "15.0 ms")
        sc3.metric.assert_called_once_with("LLM Generation", "90.0 ms")
        sc4.metric.assert_called_once_with("Citation Extraction", "5.4 ms")

        mock_st.subheader.assert_any_call("📋 Recent Query Activity")
        assert mock_st.dataframe.called
        table = mock_st.dataframe.call_args[0][0]
        assert len(table) == 2
        assert table[0]["Question"].endswith("...")
        assert len(table[0]["Question"]) == 63
        assert table[1]["Question"] == short_question
        assert table[1]["Tokens"] == "N/A"


def test_render_metrics_dashboard_empty_and_zero_stages():
    """Verify render_metrics_dashboard handles zero stage latencies and empty query activity."""
    metrics = {
        "total_queries": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "avg_total_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "avg_stage_latencies_ms": {"retrieval": 0.0, "generation": 0.0},
    }
    with patch("src.ui.components.st") as mock_st:
        m1, m2, m3, m4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [m1, m2, m3, m4]

        render_metrics_dashboard(metrics, queries=[])

        # Stage latency subheader should not be called when all values are zero
        for call in mock_st.subheader.call_args_list:
            assert "⚡ Latency by Pipeline Stage" not in call[0][0]

        mock_st.info.assert_called_once_with(
            "No queries executed yet. Ask questions in the **Codebase Chat** tab to populate telemetry."
        )


# ============================================================================
# CLI Runner Tests
# ============================================================================

def test_cli_parse_args():
    """Test CLI argument parser with defaults and overrides."""
    args = parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 8501
    assert args.api_url == "http://localhost:8000"
    assert args.headless is True

    custom = parse_args(["--host", "127.0.0.1", "--port", "9090", "--api-url", "http://backend:8000"])
    assert custom.host == "127.0.0.1"
    assert custom.port == 9090
    assert custom.api_url == "http://backend:8000"


def test_get_app_path():
    """Verify get_app_path locates existing app.py file."""
    app_path = get_app_path()
    assert app_path.exists()
    assert app_path.name == "app.py"


def test_ui_main_execution():
    """Verify ui_main builds command and calls subprocess.run."""
    with patch("subprocess.run") as mock_run:
        ui_main(["--host", "127.0.0.1", "--port", "9090", "--api-url", "http://backend:8000", "--headless"])
        assert os.environ.get("API_BASE_URL") == "http://backend:8000"
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[1:4] == ["-m", "streamlit", "run"]
        assert "--server.port" in cmd
        assert "9090" in cmd
        assert "--server.address" in cmd
        assert "127.0.0.1" in cmd
        assert "--server.headless" in cmd
        assert "true" in cmd


def test_ui_main_keyboard_interrupt():
    """Verify ui_main handles KeyboardInterrupt cleanly without traceback."""
    with patch("subprocess.run", side_effect=KeyboardInterrupt):
        ui_main([])  # Should not raise exception
