"""
Observability package for the AI Codebase Intelligence Platform.
Provides latency tracking, pipeline stage profiling, LLM token & cost monitoring,
and ingestion transparency audit logging.
"""
from __future__ import annotations

from src.observability.models import (
    StageTimer,
    PricingConfig,
    calculate_llm_cost,
    FilteredFileRecord,
    IngestionAuditLog,
    QueryTelemetry,
    MetricsSummary,
)
from src.observability.collector import (
    TelemetryCollector,
    get_telemetry_collector,
)

__all__ = [
    "StageTimer",
    "PricingConfig",
    "calculate_llm_cost",
    "FilteredFileRecord",
    "IngestionAuditLog",
    "QueryTelemetry",
    "MetricsSummary",
    "TelemetryCollector",
    "get_telemetry_collector",
]
