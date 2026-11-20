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
