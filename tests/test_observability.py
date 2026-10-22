"""
Comprehensive unit test suite for Observability & Cost Monitoring module.
Tests StageTimer, pricing models, IngestionAuditLog, TelemetryCollector,
RAGEngine profiling, FastAPI observability endpoints, and UI client methods.
Runs completely offline without network access, API keys, or GPU inference.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from src.observability.models import (
    StageTimer,
    PricingConfig,
    MODEL_PRICING,
    calculate_llm_cost,
    FilteredFileRecord,
    IngestionAuditLog,
    QueryTelemetry,
    MetricsSummary,
)
from src.observability.collector import (
    TelemetryCollector,
    get_telemetry_collector,
    _percentile,
)
from src.ingestion.filters import classify_file, audit_filter_files
from src.rag.engine import RAGEngine
from src.storage.memory_store import InMemoryVectorStore
from src.llm.fake import FakeLLMProvider
from src.registry.registry import RepoRegistry
from src.registry.models import RepositoryMetadata
from src.api.app import create_app
from src.api.dependencies import (
    get_storage,
    get_llm,
    get_registry,
    get_telemetry,
)
from src.ui.client import ApiClient, ApiClientError
from src.ui.components import (
    format_latency_pill,
    format_cost_badge,
    format_metrics,
)


# ============================================================================
# 1. StageTimer Tests
# ============================================================================

def test_stage_timer_context_manager():
    """Verify StageTimer accurately measures elapsed milliseconds in context manager."""
    with StageTimer() as timer:
        time.sleep(0.01)  # 10ms

    assert timer.elapsed_ms >= 5.0
    assert timer.elapsed_ms < 500.0


def test_stage_timer_manual_start_stop():
    """Verify manual start and stop methods."""
    timer = StageTimer()
    assert timer.stop() == 0.0  # Not started

    timer.start()
    time.sleep(0.005)
    elapsed = timer.stop()
    assert elapsed > 0.0
    assert timer.elapsed_ms == elapsed


# ============================================================================
# 2. Pricing and Cost Calculation Tests
# ============================================================================

def test_calculate_llm_cost_fake_and_zero():
    """Fake model and zero tokens should yield exactly 0.0 cost."""
    assert calculate_llm_cost("fake", 1000, 500) == 0.0
    assert calculate_llm_cost("fake-model", 5000, 2000) == 0.0
    assert calculate_llm_cost("gemini-3.8-flash", 0, 0) == 0.0
    assert calculate_llm_cost("gemini-3.8-flash", None, None) == 0.0


def test_calculate_llm_cost_gemini_flash():
    """Verify Gemini Flash pricing math: $0.15/1M prompt, $0.60/1M completion."""
    # 1,000,000 prompt tokens = $0.15
    # 1,000,000 completion tokens = $0.60
    cost = calculate_llm_cost("gemini-3.8-flash", 1_000_000, 1_000_000)
    assert pytest.approx(cost, rel=1e-5) == 0.75

    # 10,000 prompt tokens ($0.0015) + 2,000 completion tokens ($0.0012) = $0.0027
    cost_small = calculate_llm_cost("gemini-3.8-flash", 10_000, 2_000)
    assert pytest.approx(cost_small, rel=1e-5) == 0.0027


def test_calculate_llm_cost_gemini_pro():
    """Verify Gemini Pro pricing math: $1.25/1M prompt, $5.00/1M completion."""
    # 1,000,000 prompt = $1.25, 1,000,000 completion = $5.00
    cost = calculate_llm_cost("gemini-1.5-pro", 1_000_000, 1_000_000)
    assert pytest.approx(cost, rel=1e-5) == 6.25


def test_calculate_llm_cost_fallback_for_unknown_model():
    """Unknown model falls back to default fallback pricing."""
    cost = calculate_llm_cost("custom-finetuned-model", 100_000, 50_000)
    assert cost > 0.0


# ============================================================================
# 3. Ingestion Transparency & Audit Log Tests
# ============================================================================

def test_classify_file_rules():
    """Test individual file rule classification for blocklist."""
    # Kept file
    kept, reason, match = classify_file({"path": "src/main.py", "size": 1024})
    assert kept is True
    assert reason is None
    assert match is None

    # Noise directory
    kept, reason, match = classify_file({"path": "migrations/0001_init.py", "size": 500})
    assert kept is False
    assert reason == "noise_directory"
    assert match == "migrations"

    # Noise extension
    kept, reason, match = classify_file({"path": "assets/logo.png", "size": 2048})
    assert kept is False
    assert reason == "noise_extension"
    assert match == ".png"

    # Oversized file
    kept, reason, match = classify_file({"path": "big_dump.sql", "size": 600_000})
    assert kept is False
    assert reason == "file_too_large"
    assert "600000" in match


def test_audit_filter_files():
    """Test full audit filter generating IngestionAuditLog."""
    tree = [
        {"path": "app/routes.py", "size": 1500},
        {"path": "migrations/001.py", "size": 200},
        {"path": "static/img/icon.ico", "size": 5000},
        {"path": "huge.data", "size": 1_000_000},
        {"path": "app/models.py", "size": 2500},
    ]

    kept, audit = audit_filter_files(
        tree,
        repo_id="test-owner/test-repo",
        branch="main",
        commit_sha="abcdef123456",
    )

    assert len(kept) == 2
    assert [f["path"] for f in kept] == ["app/routes.py", "app/models.py"]

    assert audit.repo_id == "test-owner/test-repo"
    assert audit.branch == "main"
    assert audit.commit_sha == "abcdef123456"
    assert audit.total_scanned == 5
    assert audit.total_kept == 2
    assert audit.total_filtered == 3
    assert audit.reason_counts == {
        "noise_directory": 1,
        "noise_extension": 1,
        "file_too_large": 1,
    }

    # Serialization
    audit_dict = audit.to_dict()
    assert audit_dict["total_scanned"] == 5
    assert len(audit_dict["filtered_files"]) == 3

    # Deserialization
    reconstructed = IngestionAuditLog.from_dict(audit_dict)
    assert reconstructed.repo_id == audit.repo_id
    assert reconstructed.total_filtered == 3
    assert len(reconstructed.filtered_files) == 3

    # Markdown summary
    md = audit.to_markdown_summary()
    assert "### Ingestion Transparency Audit: `test-owner/test-repo`" in md
    assert "| `noise_directory` |" in md


# ============================================================================
# 4. TelemetryCollector Tests
# ============================================================================

def test_percentile_math():
    """Test percentile helper function."""
    assert _percentile([], 50.0) == 0.0
    assert _percentile([100.0], 95.0) == 100.0

    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p50 = _percentile(values, 50.0)
    assert 50.0 <= p50 <= 60.0
    p95 = _percentile(values, 95.0)
    assert p95 >= 90.0


def test_telemetry_collector_lifecycle():
    """Test recording queries, summary aggregation, and reset."""
    collector = TelemetryCollector(max_history=10)

    # Empty summary
    summary_empty = collector.get_summary()
    assert summary_empty.total_queries == 0
    assert summary_empty.total_cost_usd == 0.0

    # Record 3 queries
    collector.record_query(
        QueryTelemetry(
            question="What is auth?",
            model="gemini-3.8-flash",
            stage_latencies_ms={"retrieval": 50.0, "assembly_rerank": 20.0, "generation": 100.0, "citations": 10.0},
            total_duration_ms=180.0,
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            cost_usd=0.000195,
        )
    )
    collector.record_query(
        QueryTelemetry(
            question="What is database?",
            model="gemini-3.8-flash",
            stage_latencies_ms={"retrieval": 60.0, "assembly_rerank": 25.0, "generation": 120.0, "citations": 15.0},
            total_duration_ms=220.0,
            prompt_tokens=600,
            completion_tokens=250,
            total_tokens=850,
            cost_usd=0.000240,
        )
    )

    summary = collector.get_summary()
    assert summary.total_queries == 2
    assert summary.total_prompt_tokens == 1100
    assert summary.total_completion_tokens == 450
    assert summary.total_tokens == 1550
    assert summary.avg_total_latency_ms == 200.0
    assert summary.avg_stage_latencies_ms["retrieval"] == 55.0
    assert summary.total_cost_usd == pytest.approx(0.000435, rel=1e-5)

    recent = collector.get_recent_queries(limit=10)
    assert len(recent) == 2
    assert recent[0].question == "What is database?"  # newest first

    # Clear
    collector.clear()
    assert collector.get_summary().total_queries == 0
    assert len(collector.get_recent_queries()) == 0


def test_telemetry_collector_audit_storage():
    """Test storing and retrieving IngestionAuditLog records."""
    collector = TelemetryCollector()
    audit = IngestionAuditLog(
        repo_id="test/repo",
        branch="main",
        total_scanned=10,
        total_kept=8,
        total_filtered=2,
    )
    collector.record_audit(audit)

    fetched = collector.get_audit("test/repo")
    assert fetched is not None
    assert fetched.total_kept == 8
    assert collector.get_audit("nonexistent") is None


# ============================================================================
# 5. RAGEngine Observability Integration
# ============================================================================

def test_rag_engine_records_telemetry_and_cost():
    """Verify that RAGEngine records query latency breakdown and cost to collector."""
    mock_store = MagicMock()
    mock_store.query.return_value = [
        {
            "id": "c1",
            "text": "def authenticate_user(): pass",
            "path": "app/auth.py",
            "start_line": 1,
            "end_line": 5,
            "language": "python",
            "similarity": 0.85,
        }
    ]

    fake_llm = FakeLLMProvider(canned_response="Authentication is in app/auth.py:1-5.")
    collector = get_telemetry_collector()
    collector.clear()

    engine = RAGEngine(
        vector_store=mock_store,
        llm_provider=fake_llm,
        default_collection="test_col",
    )

    response = engine.answer_question("How does auth work?", collection_name="test_col")

    # Verify RAGResponse metadata enrichment
    meta = response.metadata
    assert "duration_ms" in meta
    assert meta["duration_ms"] > 0
    assert "stage_durations_ms" in meta
    stages = meta["stage_durations_ms"]
    assert "retrieval" in stages
    assert "assembly_rerank" in stages
    assert "generation" in stages
    assert "citations" in stages
    assert "estimated_cost_usd" in meta
    assert meta["estimated_cost_usd"] == 0.0  # FakeLLM is $0.0
    assert "query_id" in meta

    # Verify query was saved to global collector
    summary = collector.get_summary()
    assert summary.total_queries == 1
    assert summary.total_tokens > 0


def test_rag_engine_empty_retrieval_records_telemetry():
    """Verify empty retrieval query also records telemetry with zero cost."""
    mock_store = MagicMock()
    mock_store.query.return_value = []
    fake_llm = FakeLLMProvider()
    collector = get_telemetry_collector()
    collector.clear()

    engine = RAGEngine(
        vector_store=mock_store,
        llm_provider=fake_llm,
        default_collection="empty_col",
    )

    response = engine.answer_question("Nonexistent question", collection_name="empty_col")

    assert response.metadata["status"] == "no_context"
    assert response.metadata["duration_ms"] >= 0
    assert response.metadata["estimated_cost_usd"] == 0.0

    summary = collector.get_summary()
    assert summary.total_queries == 1


# ============================================================================
# 6. REST API Observability Endpoints Tests
# ============================================================================

@pytest.fixture
def api_test_client(tmp_path):
    """Fixture providing a test client with dependency overrides."""
    registry_file = str(tmp_path / "registry.json")
    registry = RepoRegistry(registry_file)
    vector_store = InMemoryVectorStore()
    fake_llm = FakeLLMProvider(canned_response="API Answer from app/test.py:1-10.")
    collector = TelemetryCollector()

    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_telemetry] = lambda: collector

    client = TestClient(app)
    return client, registry, collector


def test_api_metrics_endpoint(api_test_client):
    """Test GET /metrics returns aggregated summary."""
    client, registry, collector = api_test_client

    collector.record_query(
        QueryTelemetry(
            question="How does login work?",
            model="gemini-3.8-flash",
            stage_latencies_ms={"retrieval": 40.0, "generation": 110.0},
            total_duration_ms=150.0,
            prompt_tokens=400,
            completion_tokens=150,
            total_tokens=550,
            cost_usd=0.00015,
        )
    )

    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_queries"] == 1
    assert data["total_tokens"] == 550
    assert data["total_cost_usd"] == 0.00015
    assert data["avg_total_latency_ms"] == 150.0


def test_api_recent_queries_endpoint(api_test_client):
    """Test GET /metrics/queries returns recent queries list."""
    client, registry, collector = api_test_client

    collector.record_query(
        QueryTelemetry(
            question="Query 1?",
            total_duration_ms=100.0,
            cost_usd=0.0001,
        )
    )
    collector.record_query(
        QueryTelemetry(
            question="Query 2?",
            total_duration_ms=120.0,
            cost_usd=0.0002,
        )
    )

    resp = client.get("/metrics/queries?limit=10")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert items[0]["question"] == "Query 2?"  # newest first


def test_api_repo_audit_endpoint(api_test_client):
    """Test GET /repos/{owner}/{repo}/audit endpoint."""
    client, registry, collector = api_test_client

    # 1. Nonexistent repo -> 404
    resp_404 = client.get("/repos/unknown/repo/audit")
    assert resp_404.status_code == 404

    # 2. Register repo with audit log
    audit = IngestionAuditLog(
        repo_id="owner/repo",
        branch="main",
        commit_sha="1234567",
        total_scanned=20,
        total_kept=15,
        total_filtered=5,
        reason_counts={"noise_directory": 3, "noise_extension": 2},
        filtered_files=[
            FilteredFileRecord(path="migrations/1.py", reason="noise_directory", rule_match="migrations"),
            FilteredFileRecord(path="logo.png", reason="noise_extension", rule_match=".png"),
        ],
    )
    collector.record_audit(audit)

    resp_200 = client.get("/repos/owner/repo/audit")
    assert resp_200.status_code == 200
    data = resp_200.json()
    assert data["repo_id"] == "owner/repo"
    assert data["total_scanned"] == 20
    assert data["total_kept"] == 15
    assert data["total_filtered"] == 5
    assert len(data["filtered_files"]) == 2


def test_api_reset_metrics_endpoint(api_test_client):
    """Test DELETE /metrics resets counters."""
    client, registry, collector = api_test_client
    collector.record_query(QueryTelemetry(question="test"))
    assert collector.get_summary().total_queries == 1

    resp = client.delete("/metrics")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert collector.get_summary().total_queries == 0


# ============================================================================
# 7. UI Client & Presentation Formatting Tests
# ============================================================================

def test_ui_client_observability_methods():
    """Test ApiClient methods for observability."""
    client = ApiClient("http://fake-api:8000")

    # get_metrics
    with patch.object(client, "_request", return_value={"total_queries": 5, "total_cost_usd": 0.001}) as mock_r:
        m = client.get_metrics()
        assert m["total_queries"] == 5
        mock_r.assert_called_once_with("GET", "/metrics")

    # get_recent_queries
    with patch.object(client, "_request", return_value=[{"query_id": "123"}]) as mock_r:
        q = client.get_recent_queries(limit=25)
        assert len(q) == 1
        mock_r.assert_called_once_with("GET", "/metrics/queries", params={"limit": 25})

    # get_repo_audit
    with patch.object(client, "_request", return_value={"repo_id": "foo/bar", "total_scanned": 10}) as mock_r:
        a = client.get_repo_audit("foo", "bar")
        assert a["repo_id"] == "foo/bar"
        mock_r.assert_called_once_with("GET", "/repos/foo/bar/audit")

    # reset_metrics
    with patch.object(client, "_request", return_value={"success": True}) as mock_r:
        r = client.reset_metrics()
        assert r["success"] is True
        mock_r.assert_called_once_with("DELETE", "/metrics")


def test_ui_formatting_helpers():
    """Test latency, cost, and metric badge formatters."""
    # format_latency_pill
    assert format_latency_pill(None) == ""
    assert format_latency_pill(150.4) == "⏱️ 150.4 ms"
    pill = format_latency_pill(250.0, {"retrieval": 50.0, "generation": 200.0})
    assert "⏱️ 250.0 ms" in pill
    assert "Retrieval: 50.0ms" in pill

    # format_cost_badge
    assert format_cost_badge(None) == "💰 $0.00"
    assert format_cost_badge(0.0) == "💰 $0.00"
    assert format_cost_badge(0.000125) == "💰 $0.000125"

    # format_metrics with cost and duration
    meta = {
        "context_tokens": 1024,
        "max_context_tokens": 4096,
        "retrieved_count": 10,
        "selected_count": 4,
        "reranked": True,
        "duration_ms": 320.0,
        "estimated_cost_usd": 0.00035,
    }
    summary = format_metrics(meta)
    assert "Tokens: 1024/4096" in summary
    assert "Chunks: 4/10" in summary
    assert "Re-ranked: Yes" in summary
    assert "Time: 320ms" in summary
    assert "Cost: $0.000350" in summary
