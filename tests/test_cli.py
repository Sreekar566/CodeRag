"""
Unit tests for the AI Codebase Intelligence Platform Developer CLI:
- src.cli.main
- Argument parser configuration and validation
- Commands: status, repos, query, index, changes, delete
- Output modes: formatted text, JSON export
- Execution targets: local in-process and remote API client
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from src.cli.main import (
    build_parser,
    main,
    parse_repo_arg,
)
from src.registry.models import RepositoryMetadata
from src.registry.registry import RepoRegistry
from src.ui.client import ApiClientError


def test_parse_repo_arg() -> None:
    assert parse_repo_arg("pallets/flask") == ("pallets", "flask")
    assert parse_repo_arg("  miguelgrinberg/microblog  ") == ("miguelgrinberg", "microblog")

    with pytest.raises(ValueError, match="Invalid repository format"):
        parse_repo_arg("invalid-repo-name")

    with pytest.raises(ValueError, match="Invalid repository format"):
        parse_repo_arg("owner/")

    with pytest.raises(ValueError, match="Invalid repository format"):
        parse_repo_arg("/repo")


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    code = main(["--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "AI Codebase Intelligence Platform - Developer CLI" in captured.out
    assert "status" in captured.out
    assert "repos" in captured.out
    assert "query" in captured.out
    assert "index" in captured.out
    assert "changes" in captured.out
    assert "delete" in captured.out
    assert "metrics" in captured.out
    assert "audit" in captured.out


def test_cli_no_subcommand(capsys: pytest.CaptureFixture) -> None:
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "AI Codebase Intelligence Platform - Developer CLI" in captured.out


def test_cli_status_local(capsys: pytest.CaptureFixture) -> None:
    code = main(["status"])
    assert code == 0
    captured = capsys.readouterr()
    assert "AI Codebase Intelligence Platform - Status" in captured.out
    assert "Status:" in captured.out
    assert "Vector Store:" in captured.out
    assert "Registered Repos:" in captured.out
    assert "LLM Provider:" in captured.out


def test_cli_status_json(capsys: pytest.CaptureFixture) -> None:
    code = main(["status", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "ok"
    assert "vector_store" in data
    assert "collections" in data
    assert "registered_repos_count" in data
    assert "llm_provider" in data


@patch("src.ui.client.ApiClient.health")
@patch("src.ui.client.ApiClient.list_collections")
@patch("src.ui.client.ApiClient.list_repos")
def test_cli_status_remote(
    mock_repos: MagicMock,
    mock_collections: MagicMock,
    mock_health: MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    mock_health.return_value = {
        "status": "ok",
        "version": "1.2.3",
        "vector_store_type": "ChromaStore",
        "llm_provider": "GeminiLLMProvider",
        "llm_model": "gemini-3.8-flash",
    }
    mock_collections.return_value = [{"name": "repo1"}]
    mock_repos.return_value = [{"owner": "foo", "repo": "bar"}]

    code = main(["--api-url", "http://localhost:8000", "status"])
    assert code == 0
    captured = capsys.readouterr()
    assert "AI Codebase Intelligence Platform - Status" in captured.out
    assert "API Endpoint:         http://localhost:8000" in captured.out


@patch("src.ui.client.ApiClient.health", side_effect=ApiClientError("Connection refused"))
def test_cli_status_remote_error(mock_health: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["--api-url", "http://localhost:8000", "status"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error connecting to API server" in captured.err


def test_cli_repos_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    empty_reg = tmp_path / "empty_registry.json"
    reg = RepoRegistry(storage_path=str(empty_reg))
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["repos"])
    assert code == 0
    captured = capsys.readouterr()
    assert "No repositories currently registered or indexed." in captured.out


def test_cli_repos_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    reg_file = tmp_path / "test_registry.json"
    reg = RepoRegistry(storage_path=str(reg_file))
    reg.register(
        RepositoryMetadata(
            owner="octocat",
            repo="Hello-World",
            collection_name="octocat_Hello_World",
            branch="main",
            commit_sha="a1b2c3d4e5f6",
            indexed_at="2026-11-20T10:00:00Z",
            file_count=5,
            chunk_count=20,
            status="ready",
        )
    )
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["repos"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Registered Codebase Repositories" in captured.out
    assert "octocat/Hello-World" in captured.out
    assert "a1b2c3d" in captured.out
    assert "Total: 1 repositories" in captured.out


def test_cli_repos_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    reg_file = tmp_path / "test_registry.json"
    reg = RepoRegistry(storage_path=str(reg_file))
    reg.register(
        RepositoryMetadata(
            owner="pallets",
            repo="flask",
            collection_name="pallets_flask",
            branch="main",
            commit_sha="abcdef123456",
            file_count=50,
            chunk_count=250,
            status="ready",
        )
    )
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["repos", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["owner"] == "pallets"
    assert data[0]["repo"] == "flask"


@patch("src.ui.client.ApiClient.list_repos")
def test_cli_repos_remote(mock_list: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_list.return_value = [
        {
            "owner": "psf",
            "repo": "requests",
            "branch": "main",
            "commit_sha": "1234567890",
            "file_count": 30,
            "chunk_count": 120,
            "status": "ready",
            "indexed_at": "2026-11-20T11:00:00Z",
        }
    ]

    code = main(["--api-url", "http://localhost:8000", "repos"])
    assert code == 0
    captured = capsys.readouterr()
    assert "psf/requests" in captured.out


@patch("src.ui.client.ApiClient.list_repos", side_effect=ApiClientError("HTTP 500"))
def test_cli_repos_remote_error(mock_list: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["--api-url", "http://localhost:8000", "repos"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error querying API" in captured.err


def test_cli_query_local(capsys: pytest.CaptureFixture) -> None:
    code = main(["query", "What is the entrypoint of the project?", "--llm", "fake"])
    assert code == 0
    captured = capsys.readouterr()
    assert "AI Codebase Intelligence - Query Result" in captured.out
    assert "Question: What is the entrypoint of the project?" in captured.out
    assert "Model:" in captured.out
    assert "Timing:" in captured.out
    assert "Answer:" in captured.out


def test_cli_query_json(capsys: pytest.CaptureFixture) -> None:
    code = main(["query", "How does routing work?", "--llm", "fake", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["query"] == "How does routing work?"
    assert "answer" in data
    assert "metadata" in data


def test_cli_query_with_repo_and_citations(capsys: pytest.CaptureFixture) -> None:
    from src.rag.citations import Citation
    from src.rag.engine import RAGResponse

    mock_resp = RAGResponse(
        query="Explain DB connection",
        answer="The database connection is managed via SQLAlchemy pool.",
        citations=[Citation(path="app/db.py", start_line=12, end_line=30, repo="octocat/repo")],
        sources=[{"text": "mock text", "path": "app/db.py"}],
        model="fake-model",
        metadata={
            "duration_ms": 150.5,
            "estimated_cost_usd": 0.0001,
            "total_tokens": 500,
            "prompt_tokens": 400,
            "completion_tokens": 100,
            "stage_durations_ms": {"retrieval": 20.0, "generation": 120.0},
        },
    )

    with patch("src.rag.engine.RAGEngine.answer_question", return_value=mock_resp):
        code = main(["query", "Explain DB connection", "--repo", "octocat/repo", "--llm", "fake"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Explain DB connection" in captured.out
        assert "The database connection is managed via SQLAlchemy pool." in captured.out
        assert "- `app/db.py:12-30` (octocat/repo)" in captured.out


@patch("src.ui.client.ApiClient.query")
def test_cli_query_remote(mock_query: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_query.return_value = {
        "query": "Where is config loaded?",
        "answer": "Config is loaded in src/config.py via BaseSettings.",
        "citations": [{"path": "src/config.py", "start_line": 1, "end_line": 20, "repo": ""}],
        "model": "gemini-3.8-flash",
        "metadata": {
            "duration_ms": 210.0,
            "estimated_cost_usd": 0.00005,
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "total_tokens": 350,
            "stage_durations_ms": {"retrieval": 35.0, "generation": 175.0},
        },
    }

    code = main(["--api-url", "http://localhost:8000", "query", "Where is config loaded?"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Config is loaded in src/config.py" in captured.out
    assert "`src/config.py:1-20`" in captured.out


@patch("src.ui.client.ApiClient.query", side_effect=ApiClientError("Query failed"))
def test_cli_query_remote_error(mock_query: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["--api-url", "http://localhost:8000", "query", "Fail test"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error querying API" in captured.err


def test_cli_index_invalid_repo(capsys: pytest.CaptureFixture) -> None:
    code = main(["index", "not-a-repo"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Invalid repository format" in captured.err


@patch("src.storage.indexer.index_repository")
def test_cli_index_local(mock_indexer: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_indexer.return_value = {
        "branch": "main",
        "commit_sha": "1234abcd",
        "files_processed": 15,
        "chunks_stored": 45,
        "collection": "octocat_spoon_knife",
    }

    code = main(["index", "octocat/Spoon-Knife", "--branch", "main"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Repository 'octocat/Spoon-Knife' indexing complete:" in captured.out
    assert "Commit SHA:      1234abcd" in captured.out
    assert "Files Processed: 15" in captured.out
    assert "Chunks Stored:   45" in captured.out


@patch("src.storage.indexer.index_repository")
def test_cli_index_local_json(mock_indexer: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_indexer.return_value = {
        "branch": "develop",
        "commit_sha": "deadbeef",
        "files_processed": 8,
        "chunks_stored": 24,
        "collection": "test_repo",
    }

    code = main(["index", "test/repo", "--branch", "develop", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["commit_sha"] == "deadbeef"
    assert data["chunks_stored"] == 24


@patch("src.storage.indexer.index_repository", side_effect=RuntimeError("GitHub API rate limit exceeded"))
def test_cli_index_local_error(mock_indexer: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["index", "owner/repo"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error indexing repository 'owner/repo'" in captured.err


@patch("src.ui.client.ApiClient.index_repo")
def test_cli_index_remote(mock_index: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_index.return_value = {
        "branch": "main",
        "commit_sha": "abc12345",
        "filtered_files": 10,
        "chunks_stored": 30,
        "collection": "remote_repo",
    }

    code = main(["--api-url", "http://localhost:8000", "index", "remote/repo"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Repository 'remote/repo' indexing complete:" in captured.out


@patch("src.ui.client.ApiClient.index_repo", side_effect=ApiClientError("409 Conflict"))
def test_cli_index_remote_error(mock_index: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["--api-url", "http://localhost:8000", "index", "remote/repo"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error indexing repository via API" in captured.err


@patch("src.ingestion.github_client.get_latest_commit_sha", return_value="f00ba4")
@patch("src.ingestion.github_client.get_repo_tree", return_value=[{"path": "README.md", "sha": "111"}])
@patch("src.ingestion.filters.filter_files", return_value=[{"path": "README.md", "sha": "111"}])
def test_cli_changes_local(
    mock_filter: MagicMock,
    mock_tree: MagicMock,
    mock_sha: MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["changes", "foo/bar"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Pending Git Changes for 'foo/bar'" in captured.out
    assert "Remote Commit:    f00ba4" in captured.out
    assert "Added Files:      1" in captured.out


@patch("src.ingestion.github_client.get_repo_tree", side_effect=RuntimeError("Not found"))
def test_cli_changes_local_error(mock_tree: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["changes", "foo/bar"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error fetching tree from GitHub" in captured.err


@patch("src.ui.client.ApiClient.get_repo_changes")
def test_cli_changes_remote(mock_changes: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_changes.return_value = {
        "owner": "remote",
        "repo": "changes",
        "branch": "main",
        "current_commit_sha": "111111",
        "latest_commit_sha": "222222",
        "has_changes": True,
        "added_files": ["new_file.py"],
        "modified_files": [],
        "deleted_files": [],
        "unchanged_files": ["main.py"],
        "total_files": 2,
    }

    code = main(["--api-url", "http://localhost:8000", "changes", "remote/changes"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Pending Git Changes for 'remote/changes'" in captured.out
    assert "Has Changes:      True" in captured.out
    assert "+ new_file.py" in captured.out


def test_cli_delete_local_not_found(capsys: pytest.CaptureFixture) -> None:
    code = main(["delete", "nonexistent/repo", "-y"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Repository 'nonexistent/repo' not found in registry." in captured.err


def test_cli_delete_local_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    reg_file = tmp_path / "test_reg.json"
    reg = RepoRegistry(storage_path=str(reg_file))
    reg.register(
        RepositoryMetadata(
            owner="delete",
            repo="target",
            collection_name="delete_target",
            branch="main",
            commit_sha="123",
            file_count=1,
            chunk_count=1,
            status="ready",
        )
    )
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["delete", "delete/target", "-y"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Successfully deleted repository 'delete/target'." in captured.out
    assert reg.get("delete", "target") is None


def test_cli_delete_cancelled(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "n")
    code = main(["delete", "delete/target"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Deletion cancelled." in captured.out


@patch("src.ui.client.ApiClient.delete_repo")
def test_cli_delete_remote(mock_delete: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_delete.return_value = {
        "success": True,
        "message": "Repository remote/delete deleted.",
        "target": "remote/delete",
    }

    code = main(["--api-url", "http://localhost:8000", "delete", "remote/delete", "-y"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Successfully deleted repository 'remote/delete'." in captured.out


def test_cli_subprocess_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "AI Codebase Intelligence Platform - Developer CLI" in result.stdout


def test_cli_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "AI Codebase Intelligence Platform - Developer CLI" in result.stdout


# ============================================================================
# Metrics Command Tests
# ============================================================================


def test_cli_metrics_local_empty(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    code = main(["metrics"])
    assert code == 0
    captured = capsys.readouterr()
    assert "AI Codebase Intelligence - System Metrics" in captured.out
    assert "Queries Processed:    0 (Errors: 0)" in captured.out
    assert "Prompt Tokens:        0" in captured.out


def test_cli_metrics_local_with_data(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector
    from src.observability.models import QueryTelemetry

    collector = get_telemetry_collector()
    collector.clear()
    collector.record_query(
        QueryTelemetry(
            question="How does login work?",
            target_repos=["octocat/repo"],
            model="gemini-3.8-flash",
            total_duration_ms=250.0,
            prompt_tokens=300,
            completion_tokens=80,
            total_tokens=380,
            cost_usd=0.000093,
            stage_latencies_ms={"retrieval": 40.0, "generation": 200.0},
        )
    )

    code = main(["metrics"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Queries Processed:    1 (Errors: 0)" in captured.out
    assert "Prompt Tokens:        300" in captured.out
    assert "Total Tokens:         380" in captured.out
    assert "Retrieval" in captured.out


def test_cli_metrics_json(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    code = main(["metrics", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "total_queries" in data
    assert "uptime_seconds" in data
    assert "avg_total_latency_ms" in data


def test_cli_metrics_prometheus(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    code = main(["metrics", "--prometheus"])
    assert code == 0
    captured = capsys.readouterr()
    assert "# HELP coderag_queries_total" in captured.out
    assert "# TYPE coderag_queries_total counter" in captured.out
    assert "coderag_queries_total 0" in captured.out
    assert "coderag_uptime_seconds" in captured.out


def test_cli_metrics_reset(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector
    from src.observability.models import QueryTelemetry

    collector = get_telemetry_collector()
    collector.record_query(QueryTelemetry(question="test"))
    assert len(collector.get_recent_queries()) > 0

    code = main(["metrics", "--reset"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Telemetry metrics and query logs reset successfully." in captured.out
    assert len(collector.get_recent_queries()) == 0


def test_cli_metrics_queries_empty(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    code = main(["metrics", "--queries"])
    assert code == 0
    captured = capsys.readouterr()
    assert "No query telemetry records recorded yet." in captured.out


def test_cli_metrics_queries_with_data(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector
    from src.observability.models import QueryTelemetry

    collector = get_telemetry_collector()
    collector.clear()
    collector.record_query(
        QueryTelemetry(
            question="What is the app architecture?",
            target_repos=["foo/bar"],
            total_duration_ms=185.0,
            total_tokens=220,
            cost_usd=0.00005,
        )
    )

    code = main(["metrics", "--queries"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Recent Query Telemetry (showing 1 records)" in captured.out
    assert "foo/bar" in captured.out
    assert "What is the app architecture?" in captured.out


def test_cli_metrics_queries_json(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector
    from src.observability.models import QueryTelemetry

    collector = get_telemetry_collector()
    collector.clear()
    collector.record_query(QueryTelemetry(question="test query"))

    code = main(["metrics", "--queries", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["question"] == "test query"


@patch("src.ui.client.ApiClient.get_metrics")
def test_cli_metrics_remote(mock_metrics: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_metrics.return_value = {
        "total_queries": 42,
        "total_prompt_tokens": 1000,
        "total_completion_tokens": 200,
        "total_tokens": 1200,
        "total_cost_usd": 0.0005,
        "avg_total_latency_ms": 310.2,
        "p50_latency_ms": 290.0,
        "p95_latency_ms": 450.0,
        "p99_latency_ms": 520.0,
        "error_count": 1,
        "uptime_seconds": 120.0,
        "avg_stage_latencies_ms": {"retrieval": 50.0, "generation": 250.0},
    }

    code = main(["--api-url", "http://localhost:8000", "metrics"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Queries Processed:    42 (Errors: 1)" in captured.out
    assert "Total Tokens:         1,200" in captured.out
    assert "Retrieval" in captured.out


@patch("src.ui.client.ApiClient.get_prometheus_metrics")
def test_cli_metrics_remote_prometheus(mock_prom: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_prom.return_value = "coderag_queries_total 10\n"

    code = main(["--api-url", "http://localhost:8000", "metrics", "--prometheus"])
    assert code == 0
    captured = capsys.readouterr()
    assert "coderag_queries_total 10" in captured.out


@patch("src.ui.client.ApiClient.get_recent_queries")
def test_cli_metrics_remote_queries(mock_queries: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_queries.return_value = [
        {
            "timestamp": "2026-11-20T12:00:00Z",
            "total_duration_ms": 150.0,
            "total_tokens": 300,
            "cost_usd": 0.00007,
            "target_repos": ["pallets/flask"],
            "question": "How to register blueprints?",
        }
    ]

    code = main(["--api-url", "http://localhost:8000", "metrics", "--queries"])
    assert code == 0
    captured = capsys.readouterr()
    assert "pallets/flask" in captured.out
    assert "How to register blueprints?" in captured.out


@patch("src.ui.client.ApiClient.reset_metrics")
def test_cli_metrics_remote_reset(mock_reset: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_reset.return_value = {"success": True}

    code = main(["--api-url", "http://localhost:8000", "metrics", "--reset"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Telemetry metrics and query logs reset successfully." in captured.out


@patch("src.ui.client.ApiClient.get_metrics", side_effect=ApiClientError("HTTP 503"))
def test_cli_metrics_remote_error(mock_metrics: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["--api-url", "http://localhost:8000", "metrics"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error fetching metrics from API" in captured.err


# ============================================================================
# Audit Command Tests
# ============================================================================


def test_cli_audit_invalid_repo(capsys: pytest.CaptureFixture) -> None:
    code = main(["audit", "invalid_name"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Invalid repository format" in captured.err


def test_cli_audit_local_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    reg_file = tmp_path / "reg.json"
    reg = RepoRegistry(storage_path=str(reg_file))
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["audit", "nonexistent/repo"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Repository 'nonexistent/repo' is not registered or indexed." in captured.err


def test_cli_audit_local_with_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    reg_file = tmp_path / "reg.json"
    reg = RepoRegistry(storage_path=str(reg_file))
    reg.register(
        RepositoryMetadata(
            owner="audit",
            repo="test",
            collection_name="audit_test",
            branch="main",
            commit_sha="c0ffee123",
            file_count=20,
            chunk_count=60,
            status="ready",
            metadata={
                "audit_log": {
                    "repo_id": "audit/test",
                    "branch": "main",
                    "commit_sha": "c0ffee123",
                    "timestamp": "2026-11-20T10:00:00Z",
                    "total_scanned": 25,
                    "total_kept": 20,
                    "total_filtered": 5,
                    "reason_counts": {"noise_directory": 3, "file_too_large": 2},
                    "kept_files": ["app.py"],
                    "filtered_files": [
                        {
                            "path": "tests/temp.dat",
                            "reason": "noise_directory",
                            "rule_match": "tests",
                            "size_bytes": 1024,
                        }
                    ],
                }
            },
        )
    )
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["audit", "audit/test"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Ingestion Transparency Audit: audit/test" in captured.out
    assert "Total Files Scanned:  25" in captured.out
    assert "Files Kept (Indexed): 20" in captured.out
    assert "Files Filtered Out:   5" in captured.out
    assert "noise_directory" in captured.out
    assert "tests/temp.dat" in captured.out


def test_cli_audit_local_with_collector(capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector
    from src.observability.models import FilteredFileRecord, IngestionAuditLog

    collector = get_telemetry_collector()
    collector.clear()
    collector.record_audit(
        IngestionAuditLog(
            repo_id="collector/test",
            branch="master",
            commit_sha="abcdef",
            total_scanned=10,
            total_kept=8,
            total_filtered=2,
            reason_counts={"noise_extension": 2},
            kept_files=["main.py", "utils.py"],
            filtered_files=[
                FilteredFileRecord(path="logo.png", reason="noise_extension", rule_match=".png", size_bytes=5000)
            ],
        )
    )

    code = main(["audit", "collector/test"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Ingestion Transparency Audit: collector/test" in captured.out
    assert "logo.png" in captured.out


def test_cli_audit_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    reg_file = tmp_path / "reg.json"
    reg = RepoRegistry(storage_path=str(reg_file))
    reg.register(
        RepositoryMetadata(
            owner="json",
            repo="audit",
            collection_name="json_audit",
            file_count=10,
            chunk_count=30,
            status="ready",
        )
    )
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["audit", "json/audit", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["repo_id"] == "json/audit"
    assert "total_scanned" in data
    assert "filtered_files" in data


def test_cli_audit_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector

    collector = get_telemetry_collector()
    collector.clear()

    reg_file = tmp_path / "reg.json"
    reg = RepoRegistry(storage_path=str(reg_file))
    reg.register(
        RepositoryMetadata(
            owner="md",
            repo="report",
            collection_name="md_report",
            file_count=5,
            chunk_count=15,
            status="ready",
        )
    )
    monkeypatch.setattr("src.registry.registry.get_repo_registry", lambda: reg)

    code = main(["audit", "md/report", "--markdown"])
    assert code == 0
    captured = capsys.readouterr()
    assert "### Ingestion Transparency Audit: `md/report`" in captured.out


def test_cli_audit_show_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from src.observability.collector import get_telemetry_collector
    from src.observability.models import IngestionAuditLog

    collector = get_telemetry_collector()
    collector.clear()
    collector.record_audit(
        IngestionAuditLog(
            repo_id="kept/repo",
            kept_files=["src/app.py", "src/models.py"],
            total_scanned=2,
            total_kept=2,
            total_filtered=0,
        )
    )

    code = main(["audit", "kept/repo", "--show-kept"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Kept Files (2 files):" in captured.out
    assert "+ src/app.py" in captured.out
    assert "+ src/models.py" in captured.out


@patch("src.ui.client.ApiClient.get_repo_audit")
def test_cli_audit_remote(mock_audit: MagicMock, capsys: pytest.CaptureFixture) -> None:
    mock_audit.return_value = {
        "repo_id": "remote/audit",
        "branch": "main",
        "commit_sha": "999999",
        "timestamp": "2026-11-20T14:00:00Z",
        "total_scanned": 50,
        "total_kept": 45,
        "total_filtered": 5,
        "reason_counts": {"noise_directory": 5},
        "filtered_files": [
            {"path": "docs/_build/index.html", "reason": "noise_directory", "rule_match": "_build", "size_bytes": 2048}
        ],
        "kept_files": [],
    }

    code = main(["--api-url", "http://localhost:8000", "audit", "remote/audit"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Ingestion Transparency Audit: remote/audit" in captured.out
    assert "Total Files Scanned:  50" in captured.out
    assert "docs/_build/index.html" in captured.out


@patch("src.ui.client.ApiClient.get_repo_audit", side_effect=ApiClientError("404 Not Found"))
def test_cli_audit_remote_error(mock_audit: MagicMock, capsys: pytest.CaptureFixture) -> None:
    code = main(["--api-url", "http://localhost:8000", "audit", "remote/audit"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error fetching audit log from API" in captured.err
