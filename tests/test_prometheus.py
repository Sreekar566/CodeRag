"""
Unit tests for Prometheus metrics exposition and content negotiation.
Tests format_prometheus_metrics formatting, API endpoint, query parameter format,
and Accept header content negotiation.
"""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_registry, get_storage, get_telemetry
from src.observability.collector import TelemetryCollector
from src.observability.models import MetricsSummary, QueryTelemetry
from src.observability.prometheus import format_prometheus_metrics
from src.registry.registry import RepoRegistry
from src.storage.memory_store import InMemoryVectorStore
from src.ui.client import ApiClient, ApiClientError


def test_format_prometheus_metrics_empty() -> None:
    """Verify empty/default metrics summary formats into valid Prometheus text."""
    summary = MetricsSummary(uptime_seconds=12.5)
    output = format_prometheus_metrics(summary)

    assert "# HELP coderag_queries_total" in output
    assert "# TYPE coderag_queries_total counter" in output
    assert "coderag_queries_total 0" in output

    assert "# HELP coderag_query_errors_total" in output
    assert "coderag_query_errors_total 0" in output

    assert 'coderag_tokens_total{type="prompt"} 0' in output
    assert 'coderag_tokens_total{type="completion"} 0' in output
    assert 'coderag_tokens_total{type="total"} 0' in output

    assert "coderag_cost_usd_total 0.000000" in output
    assert "coderag_uptime_seconds 12.5" in output

    assert 'coderag_query_latency_ms{quantile="0.5"} 0.00' in output
    assert 'coderag_stage_latency_ms{stage="retrieval"} 0.00' in output
    assert output.endswith("\n")


def test_format_prometheus_metrics_populated() -> None:
    """Verify populated metrics summary and extra gauges format correctly."""
    summary = MetricsSummary(
        total_queries=15,
        total_prompt_tokens=3200,
        total_completion_tokens=800,
        total_tokens=4000,
        total_cost_usd=0.001250,
        avg_total_latency_ms=250.45,
        avg_stage_latencies_ms={
            "retrieval": 45.2,
            "assembly_rerank": 15.1,
            "generation": 180.0,
            "citations": 10.15,
        },
        p50_latency_ms=220.0,
        p95_latency_ms=450.0,
        p99_latency_ms=600.0,
        error_count=1,
        uptime_seconds=3600.0,
    )
    extra = {
        "coderag_indexed_repos_total": 3,
        "coderag_collections_total": 4,
    }

    output = format_prometheus_metrics(summary, extra_gauges=extra)

    assert "coderag_queries_total 15" in output
    assert "coderag_query_errors_total 1" in output
    assert 'coderag_tokens_total{type="prompt"} 3200' in output
    assert 'coderag_tokens_total{type="completion"} 800' in output
    assert 'coderag_tokens_total{type="total"} 4000' in output
    assert "coderag_cost_usd_total 0.001250" in output
    assert "coderag_uptime_seconds 3600.0" in output
    assert "coderag_latency_avg_ms 250.45" in output
    assert 'coderag_query_latency_ms{quantile="0.5"} 220.00' in output
    assert 'coderag_query_latency_ms{quantile="0.95"} 450.00' in output
    assert 'coderag_query_latency_ms{quantile="0.99"} 600.00' in output
    assert 'coderag_stage_latency_ms{stage="retrieval"} 45.20' in output
    assert 'coderag_stage_latency_ms{stage="generation"} 180.00' in output
    assert "coderag_indexed_repos_total 3" in output
    assert "coderag_collections_total 4" in output


@pytest.fixture
def prom_client(tmp_path):
    """Fixture providing an isolated FastAPI TestClient with dummy data."""
    registry_file = str(tmp_path / "test_reg.json")
    registry = RepoRegistry(registry_file)
    storage = InMemoryVectorStore()
    collector = TelemetryCollector()

    collector.record_query(
        QueryTelemetry(
            question="Where is login?",
            model="gemini-3.8-flash",
            stage_latencies_ms={"retrieval": 30.0, "generation": 100.0},
            total_duration_ms=130.0,
            prompt_tokens=500,
            completion_tokens=100,
            total_tokens=600,
            cost_usd=0.000135,
        )
    )

    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_telemetry] = lambda: collector

    client = TestClient(app)
    return client, collector


def test_api_prometheus_endpoint(prom_client):
    """Verify GET /metrics/prometheus returns text/plain Prometheus format."""
    client, _ = prom_client
    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    assert "coderag_queries_total 1" in resp.text
    assert 'coderag_tokens_total{type="prompt"} 500' in resp.text
    assert "coderag_indexed_repos_total 0" in resp.text


def test_api_metrics_default_returns_json(prom_client):
    """Verify GET /metrics with default headers returns JSON schema."""
    client, _ = prom_client
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "")
    data = resp.json()
    assert data["total_queries"] == 1
    assert data["total_tokens"] == 600


def test_api_metrics_format_prometheus_query_param(prom_client):
    """Verify GET /metrics?format=prometheus returns Prometheus text."""
    client, _ = prom_client
    resp = client.get("/metrics?format=prometheus")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    assert "coderag_queries_total 1" in resp.text


def test_api_metrics_accept_header_content_negotiation(prom_client):
    """Verify GET /metrics with Accept: text/plain returns Prometheus text."""
    client, _ = prom_client
    resp = client.get("/metrics", headers={"Accept": "text/plain"})
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    assert "coderag_queries_total 1" in resp.text


def test_api_client_get_prometheus_metrics(prom_client, monkeypatch):
    """Verify ApiClient.get_prometheus_metrics retrieves text correctly."""
    client, _ = prom_client
    api_cli = ApiClient(base_url="http://testserver")
    monkeypatch.setattr(api_cli, "session", client)

    text = api_cli.get_prometheus_metrics()
    assert isinstance(text, str)
    assert "coderag_queries_total 1" in text
