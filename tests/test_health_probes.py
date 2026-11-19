"""
Unit tests for Kubernetes/container orchestration liveness and readiness health probes.
Tests /health/live, /health/ready, diagnostic subsystem reporting, and 503 degradation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_registry, get_storage
from src.registry.registry import RepoRegistry
from src.storage.memory_store import InMemoryVectorStore
from src.ui.client import ApiClient


@pytest.fixture
def health_probe_client(tmp_path):
    """Fixture providing isolated FastAPI TestClient with in-memory dependencies."""
    registry_file = str(tmp_path / "probe_reg.json")
    registry = RepoRegistry(registry_file)
    storage = InMemoryVectorStore()

    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: storage

    client = TestClient(app)
    return client, registry, storage


def test_liveness_probe_returns_200_alive(health_probe_client):
    """Verify /health/live and /healthz/live return 200 with status='alive'."""
    client, _, _ = health_probe_client

    for path in ("/health/live", "/healthz/live"):
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert "timestamp" in data


def test_readiness_probe_returns_200_when_healthy(health_probe_client):
    """Verify /health/ready returns 200 when storage, registry, and filesystem are healthy."""
    client, _, _ = health_probe_client

    for path in ("/health/ready", "/healthz/ready"):
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        checks = data["checks"]
        assert checks["storage"]["status"] == "healthy"
        assert checks["storage"]["details"]["type"] == "InMemoryVectorStore"
        assert checks["registry"]["status"] == "healthy"
        assert checks["persistence"]["status"] == "healthy"


def test_readiness_probe_returns_503_on_storage_failure(health_probe_client):
    """Verify /health/ready returns 503 when vector storage is unavailable."""
    client, _, storage = health_probe_client

    with patch.object(storage, "list_collections", side_effect=RuntimeError("Chroma disconnected")):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["storage"]["status"] == "unhealthy"
        assert "Chroma disconnected" in data["checks"]["storage"]["error"]


def test_readiness_probe_returns_503_on_registry_failure(health_probe_client):
    """Verify /health/ready returns 503 when registry metadata cannot be read."""
    client, registry, _ = health_probe_client

    with patch.object(registry, "list_repos", side_effect=IOError("Corrupt registry JSON")):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["registry"]["status"] == "unhealthy"
        assert "Corrupt registry JSON" in data["checks"]["registry"]["error"]


def test_readiness_probe_returns_503_on_unwritable_persistence(health_probe_client):
    """Verify /health/ready returns 503 when the persistence directory is read-only."""
    client, _, _ = health_probe_client

    with patch("os.access", return_value=False):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["persistence"]["status"] == "unhealthy"
        assert "not writable" in data["checks"]["persistence"]["error"]


def test_api_client_health_probes(health_probe_client, monkeypatch):
    """Verify ApiClient methods for liveness and readiness."""
    client, _, _ = health_probe_client
    api_cli = ApiClient(base_url="http://testserver")
    monkeypatch.setattr(api_cli, "session", client)

    live_res = api_cli.check_liveness()
    assert live_res["status"] == "alive"

    ready_res = api_cli.check_readiness()
    assert ready_res["status"] == "ready"
