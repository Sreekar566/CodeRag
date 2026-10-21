"""
Thread-safe telemetry collector storing runtime metrics, query performance logs,
and repository ingestion audit records.
"""
from __future__ import annotations

from collections import deque
import logging
import math
import threading
import time
from typing import Any

from src.observability.models import IngestionAuditLog, MetricsSummary, QueryTelemetry

logger = logging.getLogger(__name__)


def _percentile(values: list[float], percent: float) -> float:
    """Calculate percentile from a sorted list of floats."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (percent / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


class TelemetryCollector:
    """
    Central in-memory telemetry and observability metrics collector.
    Thread-safe for concurrent API requests.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self._queries: deque[QueryTelemetry] = deque(maxlen=max_history)
        self._audits: dict[str, IngestionAuditLog] = {}
        self._start_time = time.time()
        self._lock = threading.Lock()

    def record_query(self, telemetry: QueryTelemetry) -> None:
        """Record a query telemetry event."""
        with self._lock:
            self._queries.append(telemetry)

    def record_audit(self, audit: IngestionAuditLog) -> None:
        """Store an ingestion audit log for a repository."""
        with self._lock:
            self._audits[audit.repo_id] = audit

    def get_audit(self, repo_id: str) -> IngestionAuditLog | None:
        """Retrieve the ingestion audit log for a repository."""
        with self._lock:
            return self._audits.get(repo_id)

    def list_audits(self) -> dict[str, IngestionAuditLog]:
        """List all stored ingestion audit logs."""
        with self._lock:
            return dict(self._audits)

    def get_recent_queries(self, limit: int = 50) -> list[QueryTelemetry]:
        """Get the most recent query telemetry events (newest first)."""
        with self._lock:
            items = list(self._queries)
        items.reverse()
        return items[:limit]

    def get_summary(self) -> MetricsSummary:
        """Compute aggregated metrics summary across all recorded queries."""
        with self._lock:
            queries = list(self._queries)
            uptime = time.time() - self._start_time

        if not queries:
            return MetricsSummary(
                total_queries=0,
                uptime_seconds=uptime,
            )

        total_queries = len(queries)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost_usd = 0.0
        error_count = 0

        durations: list[float] = []
        stage_accumulators: dict[str, list[float]] = {
            "retrieval": [],
            "assembly_rerank": [],
            "generation": [],
            "citations": [],
        }

        for q in queries:
            if q.error:
                error_count += 1
            if q.prompt_tokens is not None:
                total_prompt_tokens += q.prompt_tokens
            if q.completion_tokens is not None:
                total_completion_tokens += q.completion_tokens
            if q.total_tokens is not None:
                total_tokens += q.total_tokens
            total_cost_usd += q.cost_usd
            durations.append(q.total_duration_ms)

            for stage, ms in q.stage_latencies_ms.items():
                if stage not in stage_accumulators:
                    stage_accumulators[stage] = []
                stage_accumulators[stage].append(ms)

        durations.sort()
        avg_total_latency = sum(durations) / len(durations) if durations else 0.0
        avg_stages = {
            stage: (sum(vals) / len(vals)) if vals else 0.0
            for stage, vals in stage_accumulators.items()
        }

        p50 = _percentile(durations, 50.0)
        p95 = _percentile(durations, 95.0)
        p99 = _percentile(durations, 99.0)

        return MetricsSummary(
            total_queries=total_queries,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            avg_total_latency_ms=avg_total_latency,
            avg_stage_latencies_ms=avg_stages,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            error_count=error_count,
            uptime_seconds=uptime,
        )

    def clear(self) -> None:
        """Clear query history and audits (primarily for unit test isolation)."""
        with self._lock:
            self._queries.clear()
            self._audits.clear()
            self._start_time = time.time()


# Global singleton instance
_GLOBAL_COLLECTOR: TelemetryCollector | None = None
_COLLECTOR_LOCK = threading.Lock()


def get_telemetry_collector() -> TelemetryCollector:
    """Retrieve or create the global singleton TelemetryCollector."""
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is None:
        with _COLLECTOR_LOCK:
            if _GLOBAL_COLLECTOR is None:
                _GLOBAL_COLLECTOR = TelemetryCollector()
    return _GLOBAL_COLLECTOR
