"""
Comprehensive unit tests for FastAPI backend API endpoints.
Tests run fully offline using InMemoryVectorStore, FakeLLMProvider, and mocked GitHub API.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_registry, get_storage, get_llm
from src.storage.memory_store import InMemoryVectorStore
from src.registry.registry import RepoRegistry
from src.llm.fake import FakeLLMProvider


@pytest.fixture
def mock_embedder():
    """Mock the embedding functions to return deterministic unit vectors offline."""
    dummy_vec = [0.1] * 768
    with patch("src.embedding.embedder.embed_chunks") as mock_chunks, \
         patch("src.embedding.embedder.embed_query") as mock_query, \
         patch("src.embedding.embedder.embed_texts") as mock_texts, \
         patch("src.registry.incremental.embed_chunks") as mock_incr_chunks, \
         patch("src.storage.indexer.embed_chunks") as mock_idx_chunks:

        def _fake_embed_chunks(chunks, *args, **kwargs):
            return [{**c, "embedding": list(dummy_vec)} for c in chunks]

        mock_chunks.side_effect = _fake_embed_chunks
        mock_incr_chunks.side_effect = _fake_embed_chunks
        mock_idx_chunks.side_effect = _fake_embed_chunks
        mock_query.return_value = list(dummy_vec)
        mock_texts.return_value = [list(dummy_vec)]
        yield mock_chunks, mock_query, mock_texts


@pytest.fixture
def api_client(tmp_path, mock_embedder):
    """
    Constructs a TestClient with isolated in-memory dependencies and temporary storage.
    """
    registry_file = str(tmp_path / "test_api_registry.json")
    registry = RepoRegistry(storage_path=registry_file)
    storage = InMemoryVectorStore(
        default_collection="test_default",
        embed_fn=lambda q: [0.1] * 768,
    )
    fake_llm = FakeLLMProvider(
        canned_response="Authentication is handled by login() in app/auth.py:10-25."
    )

    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_llm] = lambda: fake_llm

    client = TestClient(app)
    client.app_registry = registry
    client.app_storage = storage
    client.app_llm = fake_llm

    yield client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_github():
    """Mock GitHub client functions for indexing."""
    with patch("src.registry.incremental.get_latest_commit_sha") as mock_sha, \
         patch("src.registry.incremental.get_repo_tree") as mock_tree, \
         patch("src.registry.incremental.filter_files") as mock_filter, \
         patch("src.registry.incremental.get_file_content") as mock_content, \
         patch("src.ingestion.github_client.get_default_branch") as mock_branch, \
         patch("src.api.routes.repos.get_latest_commit_sha") as mock_route_sha, \
         patch("src.api.routes.repos.get_repo_tree") as mock_route_tree, \
         patch("src.api.routes.repos.filter_files") as mock_route_filter:

        mock_sha.return_value = "commit_sha_123"
        mock_route_sha.return_value = "commit_sha_123"
        mock_branch.return_value = "main"

        sample_tree = [
            {"path": "app/auth.py", "type": "blob", "sha": "sha_auth_1"},
            {"path": "app/routes.py", "type": "blob", "sha": "sha_routes_1"},
        ]
        mock_tree.return_value = sample_tree
        mock_filter.return_value = sample_tree
        mock_route_tree.return_value = sample_tree
        mock_route_filter.return_value = sample_tree

        def _get_content(owner, repo, path, token=None):
            if path == "app/auth.py":
                return (
                    "# Auth module\n"
                    "def login(username, password):\n"
                    "    \"\"\"Authenticate user.\"\"\"\n"
                    "    if username == 'admin':\n"
                    "        return True\n"
                    "    return False\n"
                )
            return "def index():\n    return 'Hello World'\n"

        mock_content.side_effect = _get_content

        yield {
            "sha": mock_sha,
            "tree": mock_tree,
            "filter": mock_filter,
            "content": mock_content,
        }


# ============================================================================
# Root and Health Tests
# ============================================================================

def test_api_root(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data
    assert "/health" in data["endpoints"].values()


def test_api_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["vector_store_type"] == "InMemoryVectorStore"
    assert data["llm_provider"] == "FakeLLMProvider"
    assert data["indexed_repos_count"] == 0
    assert data["collections_count"] == 0


# ============================================================================
# Repository CRUD Tests
# ============================================================================

def test_repos_list_empty(api_client):
    response = api_client.get("/repos")
    assert response.status_code == 200
    data = response.json()
    assert data["repos"] == []
    assert data["total"] == 0


def test_get_repo_not_found(api_client):
    response = api_client.get("/repos/octocat/hello-world")
    assert response.status_code == 404
    assert "not registered" in response.json()["detail"].lower()


def test_delete_repo_not_found(api_client):
    response = api_client.delete("/repos/octocat/hello-world")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ============================================================================
# Indexing Endpoint Tests
# ============================================================================

def test_index_repo_sync(api_client, mock_github):
    payload = {
        "owner": "testorg",
        "repo": "testrepo",
        "branch": "main",
    }
    response = api_client.post("/index", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["owner"] == "testorg"
    assert data["repo"] == "testrepo"
    assert data["status"] == "created"
    assert data["files_processed"] == 2
    assert data["chunks_stored"] > 0
    assert data["commit_sha"] == "commit_sha_123"
    assert data["background_task"] is False

    # Verify repository was registered
    repo_res = api_client.get("/repos/testorg/testrepo")
    assert repo_res.status_code == 200
    repo_data = repo_res.json()
    assert repo_data["owner"] == "testorg"
    assert repo_data["file_count"] == 2
    assert repo_data["chunk_count"] > 0
    assert repo_data["status"] == "ready"

    # Verify list repos
    list_res = api_client.get("/repos")
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 1

    # Verify collection exists in storage
    cols_res = api_client.get("/collections")
    assert cols_res.status_code == 200
    assert cols_res.json()["total"] >= 1


def test_index_repo_unchanged(api_client, mock_github):
    # Index once
    api_client.post("/index", json={"owner": "testorg", "repo": "testrepo"})

    # Index second time with unchanged commit SHA
    response = api_client.post("/index", json={"owner": "testorg", "repo": "testrepo"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unchanged"
    assert data["files_processed"] == 0
    assert data["chunks_stored"] == 0


def test_index_repo_force(api_client, mock_github):
    # Index once
    api_client.post("/index", json={"owner": "testorg", "repo": "testrepo"})

    # Index again with force=True
    response = api_client.post(
        "/index",
        json={"owner": "testorg", "repo": "testrepo", "force": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("created", "updated")
    assert data["files_processed"] == 2


def test_index_repo_validation_error(api_client):
    # Empty owner
    response = api_client.post("/index", json={"owner": "", "repo": "valid"})
    assert response.status_code == 422

    # Missing repo
    response = api_client.post("/index", json={"owner": "valid"})
    assert response.status_code == 422


def test_index_repo_conflict(api_client):
    # Simulate an active indexing lock
    locked = api_client.app_registry.acquire_repo_lock("testorg", "lockedrepo", blocking=False)
    assert locked is True

    try:
        response = api_client.post(
            "/index",
            json={"owner": "testorg", "repo": "lockedrepo"},
        )
        assert response.status_code == 409
        assert "currently being indexed" in response.json()["detail"]
    finally:
        api_client.app_registry.release_repo_lock("testorg", "lockedrepo")


def test_index_repo_not_found_on_github(api_client):
    with patch("src.registry.incremental.get_repo_tree", side_effect=Exception("404 Client Error: Not Found")):
        response = api_client.post(
            "/index",
            json={"owner": "nonexistent", "repo": "ghostrepo"},
        )
        assert response.status_code == 404
        assert "not found on github" in response.json()["detail"].lower()


def test_index_repo_background(api_client, mock_github):
    payload = {
        "owner": "testorg",
        "repo": "bgrepo",
        "background": True,
    }
    response = api_client.post("/index", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["background_task"] is True
    assert data["status"] == "indexing"
    assert "initiated in the background" in data["message"]


# ============================================================================
# Query Endpoint Tests
# ============================================================================

def test_query_codebase(api_client, mock_github):
    # Index repo first
    api_client.post("/index", json={"owner": "testorg", "repo": "testrepo"})

    query_payload = {
        "question": "How does login authentication work?",
        "repo": "testorg/testrepo",
        "top_k": 5,
        "rerank": True,
    }
    response = api_client.post("/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "How does login authentication work?"
    assert "login" in data["answer"].lower()
    assert "Citations:" in data["formatted_answer"]
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["path"] == "app/auth.py"
    assert data["citations"][0]["start_line"] == 10
    assert len(data["sources"]) >= 1
    assert data["model"] == "fake-model"
    assert "context_tokens" in data["metadata"]


def test_query_multi_repo(api_client, mock_github):
    # Index two repositories
    api_client.post("/index", json={"owner": "org1", "repo": "repo1"})
    api_client.post("/index", json={"owner": "org2", "repo": "repo2"})

    query_payload = {
        "question": "Where is user authentication defined?",
        "repos": ["org1/repo1", "org2/repo2"],
        "top_k": 5,
    }
    response = api_client.post("/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] != ""
    assert len(data["sources"]) >= 1


def test_query_empty_question(api_client):
    response = api_client.post("/query", json={"question": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


def test_query_unindexed_repo(api_client):
    response = api_client.post(
        "/query",
        json={"question": "How does this work?", "repo": "unindexed/repo"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "No relevant code context was found" in data["answer"]
    assert data["citations"] == []
    assert data["sources"] == []


# ============================================================================
# Collections & Changeset Endpoint Tests
# ============================================================================

def test_collections_crud(api_client, mock_github):
    # Index a repository to populate a collection
    api_client.post("/index", json={"owner": "testorg", "repo": "testrepo"})
    col_name = "testorg_testrepo"

    # List collections
    list_res = api_client.get("/collections")
    assert list_res.status_code == 200
    collections = list_res.json()["collections"]
    assert any(c["name"] == col_name for c in collections)

    # Get collection details
    detail_res = api_client.get(f"/collections/{col_name}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["name"] == col_name
    assert detail_data["chunk_count"] > 0
    assert len(detail_data["sample_chunks"]) > 0

    # Get nonexistent collection
    missing_res = api_client.get("/collections/nonexistent_col")
    assert missing_res.status_code == 404

    # Delete collection
    del_res = api_client.delete(f"/collections/{col_name}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Delete nonexistent collection
    del_missing = api_client.delete("/collections/nonexistent_col")
    assert del_missing.status_code == 404


def test_preview_repo_changes(api_client, mock_github):
    # Index repo first
    api_client.post("/index", json={"owner": "testorg", "repo": "testrepo"})

    # Preview changes
    res = api_client.get("/repos/testorg/testrepo/changes")
    assert res.status_code == 200
    data = res.json()
    assert data["owner"] == "testorg"
    assert data["repo"] == "testrepo"
    assert data["has_changes"] is False
    assert len(data["unchanged_files"]) == 2


def test_delete_repository(api_client, mock_github):
    # Index repo
    api_client.post("/index", json={"owner": "testorg", "repo": "testrepo"})
    assert api_client.app_registry.exists("testorg", "testrepo")

    # Delete repo
    del_res = api_client.delete("/repos/testorg/testrepo")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True
    assert "testorg/testrepo" in del_res.json()["message"]

    # Verify repo no longer exists
    assert not api_client.app_registry.exists("testorg", "testrepo")
    get_res = api_client.get("/repos/testorg/testrepo")
    assert get_res.status_code == 404


def test_cors_preflight(api_client):
    headers = {
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = api_client.options("/query", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"
