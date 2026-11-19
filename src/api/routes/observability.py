"""
Observability and cost monitoring API endpoints.
Provides system metrics, pipeline latency profiling, token spend,
Prometheus metrics exposition, and ingestion audit transparency.
"""
from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from src.api.dependencies import get_registry, get_storage, get_telemetry
from src.api.models import (
    DeleteResponse,
    FilteredFileResponse,
    IngestionAuditResponse,
    MetricsSummaryResponse,
    QueryTelemetryResponse,
)
from src.observability.collector import TelemetryCollector
from src.observability.models import IngestionAuditLog
from src.observability.prometheus import format_prometheus_metrics
from src.registry.registry import RepoRegistry
from src.storage.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Observability"])

PROMETHEUS_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _build_prometheus_response(
    collector: TelemetryCollector,
    registry: RepoRegistry,
    storage: VectorStore,
) -> Response:
    """Format and wrap current metrics into Prometheus text response."""
    summary = collector.get_summary()
    extra_gauges: dict[str, float | int] = {}
    try:
        extra_gauges["coderag_indexed_repos_total"] = len(registry.list_repos())
    except Exception:
        pass
    try:
        extra_gauges["coderag_collections_total"] = len(storage.list_collections())
    except Exception:
        pass

    body = format_prometheus_metrics(summary, extra_gauges=extra_gauges)
    return Response(content=body, media_type=PROMETHEUS_MEDIA_TYPE)


@router.get(
    "/metrics/prometheus",
    summary="Get system metrics in Prometheus exposition format",
    response_class=Response,
)
def get_prometheus_metrics(
    collector: TelemetryCollector = Depends(get_telemetry),
    registry: RepoRegistry = Depends(get_registry),
    storage: VectorStore = Depends(get_storage),
) -> Response:
    """
    Returns application performance counters, query latencies, token consumption,
    and estimated cost in official Prometheus text exposition format (version 0.0.4).
    """
    return _build_prometheus_response(collector, registry, storage)


@router.get(
    "/metrics",
    summary="Get aggregated system telemetry, latency percentiles, and LLM cost metrics",
    responses={
        200: {
            "content": {
                "application/json": {},
                "text/plain; version=0.0.4; charset=utf-8": {},
            },
            "description": "System telemetry summary in JSON or Prometheus text format.",
        }
    },
)
def get_metrics_summary(
    request: Request,
    format: str | None = Query(
        default=None,
        description="Response format: 'json' (default) or 'prometheus'",
    ),
    collector: TelemetryCollector = Depends(get_telemetry),
    registry: RepoRegistry = Depends(get_registry),
    storage: VectorStore = Depends(get_storage),
) -> Any:
    """
    Returns aggregated metrics across all serviced queries including:
    - Total queries serviced and error counts
    - Prompt, completion, and total token usage
    - Estimated LLM API cost in USD
    - Overall average latency and per-stage timing (retrieval, assembly, generation, citations)
    - p50, p95, and p99 latency percentiles

    Supports Prometheus exposition format when requested via `format=prometheus` query parameter
    or `Accept: text/plain` header.
    """
    accept_header = request.headers.get("accept", "")
    wants_prometheus = (
        format == "prometheus"
        or (
            format is None
            and "text/plain" in accept_header
            and "application/json" not in accept_header
            and "*/*" not in accept_header
        )
    )

    if wants_prometheus:
        return _build_prometheus_response(collector, registry, storage)

    summary = collector.get_summary()
    return MetricsSummaryResponse(**summary.to_dict())


@router.get(
    "/metrics/queries",
    response_model=list[QueryTelemetryResponse],
    summary="Get recent query telemetry and cost logs",
)
def get_recent_queries(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum query logs to return"),
    collector: TelemetryCollector = Depends(get_telemetry),
) -> list[QueryTelemetryResponse]:
    """
    Returns rolling telemetry records for recently executed RAG queries (newest first).
    """
    queries = collector.get_recent_queries(limit=limit)
    return [
        QueryTelemetryResponse(
            query_id=q.query_id,
            timestamp=q.timestamp,
            question=q.question,
            target_repos=q.target_repos,
            model=q.model,
            stage_latencies_ms=q.stage_latencies_ms,
            total_duration_ms=q.total_duration_ms,
            prompt_tokens=q.prompt_tokens,
            completion_tokens=q.completion_tokens,
            total_tokens=q.total_tokens,
            context_tokens=q.context_tokens,
            cost_usd=q.cost_usd,
            retrieved_count=q.retrieved_count,
            selected_count=q.selected_count,
            citations_count=q.citations_count,
            error=q.error,
        )
        for q in queries
    ]


@router.get(
    "/repos/{owner}/{repo}/audit",
    response_model=IngestionAuditResponse,
    summary="Get ingestion transparency audit log for a repository",
)
def get_repo_audit(
    owner: str,
    repo: str,
    collector: TelemetryCollector = Depends(get_telemetry),
    registry: RepoRegistry = Depends(get_registry),
) -> IngestionAuditResponse:
    """
    Returns the file-filtering audit log for an indexed repository, showing which files
    were kept or filtered out and the corresponding blocklist rule matched.
    """
    repo_id = f"{owner}/{repo}"

    # 1. Check in-memory collector
    audit = collector.get_audit(repo_id)

    # 2. Check persistent registry metadata if not in collector
    if audit is None:
        meta = registry.get(owner, repo)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository '{owner}/{repo}' is not registered or indexed.",
            )
        audit_data = meta.metadata.get("audit_log") if meta.metadata else None
        if audit_data:
            audit = IngestionAuditLog.from_dict(audit_data)
        else:
            # Synthetic response for legacy registered repo without audit log
            audit = IngestionAuditLog(
                repo_id=repo_id,
                branch=meta.branch,
                commit_sha=meta.commit_sha,
                total_scanned=meta.file_count,
                total_kept=meta.file_count,
                total_filtered=0,
                reason_counts={},
                kept_files=[],
                filtered_files=[],
            )

    filtered_res = [
        FilteredFileResponse(
            path=f.path,
            reason=f.reason,
            rule_match=f.rule_match,
            size_bytes=f.size_bytes,
        )
        for f in audit.filtered_files
    ]

    return IngestionAuditResponse(
        repo_id=audit.repo_id,
        branch=audit.branch,
        commit_sha=audit.commit_sha,
        timestamp=audit.timestamp,
        total_scanned=audit.total_scanned,
        total_kept=audit.total_kept,
        total_filtered=audit.total_filtered,
        reason_counts=audit.reason_counts,
        kept_files=audit.kept_files,
        filtered_files=filtered_res,
    )


@router.delete(
    "/metrics",
    response_model=DeleteResponse,
    summary="Reset in-memory telemetry records and query logs",
)
def reset_metrics(
    collector: TelemetryCollector = Depends(get_telemetry),
) -> DeleteResponse:
    """Reset all in-memory query telemetry and performance counters."""
    collector.clear()
    return DeleteResponse(
        success=True,
        message="Telemetry metrics and query logs reset successfully.",
        target="metrics",
    )
