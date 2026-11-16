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
from src.observability.logging import (
    DEFAULT_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    JsonLogFormatter,
    RequestIdFilter,
    get_current_request_id,
    reset_current_request_id,
    set_current_request_id,
    setup_logging,
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
    "DEFAULT_DATE_FORMAT",
    "DEFAULT_LOG_FORMAT",
    "JsonLogFormatter",
    "RequestIdFilter",
    "get_current_request_id",
    "set_current_request_id",
    "reset_current_request_id",
    "setup_logging",
]
