"""
Comprehensive unit tests for the Streamlit UI components, API client, and CLI runner.
All tests run offline without network access or running backend servers.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.ui.client import ApiClient, ApiClientError, ApiConnectionError
from src.ui.components import (
    format_citation,
    format_metrics,
    format_timestamp,
    get_status_badge,
    format_repo_label,
    format_changes_summary,
)
from src.ui.main import parse_args, get_app_path


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
    """Test request timeout raises ApiConnectionError."""
    client = ApiClient("http://fake-api:8000")
    with patch.object(
        client.session,
        "request",
        side_effect=requests.exceptions.Timeout("Request timed out"),
    ):
        with pytest.raises(ApiConnectionError) as exc_info:
            client.health()
        assert "Failed to connect to backend server" in str(exc_info.value)


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


def test_format_timestamp():
    """Test format_timestamp date cleaner."""
    assert format_timestamp(None) == "N/A"
    assert format_timestamp("") == "N/A"
    assert format_timestamp("2026-10-21T00:00:00.123456") == "2026-10-21 00:00:00 UTC"


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
    """Test format_changes_summary."""
    changes = {
        "added": ["a.py", "b.py"],
        "modified": ["c.py"],
        "deleted": [],
        "unchanged": ["d.py", "e.py"],
    }
    summary = format_changes_summary(changes)
    assert summary == "+2 added, ~1 modified, -0 deleted, =2 unchanged"


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
