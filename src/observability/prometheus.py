"""
Prometheus metrics exporter for the AI Codebase Intelligence Platform.
Converts system telemetry, pipeline latencies, token consumption, and cost
into official Prometheus text exposition format (version 0.0.4).
"""
from __future__ import annotations

from typing import Any
from src.observability.models import MetricsSummary


def format_prometheus_metrics(
    summary: MetricsSummary,
    extra_gauges: dict[str, float | int] | None = None,
) -> str:
    """
    Format MetricsSummary and optional extra metrics into Prometheus text format v0.0.4.

    Args:
        summary: Aggregated MetricsSummary instance from TelemetryCollector.
        extra_gauges: Optional dictionary of additional gauge metric names and values.

    Returns:
        String containing Prometheus-compliant metric definitions and samples.
    """
    lines: list[str] = []

    # 1. Total Queries Counter
    lines.append("# HELP coderag_queries_total Total number of RAG queries processed.")
    lines.append("# TYPE coderag_queries_total counter")
    lines.append(f"coderag_queries_total {summary.total_queries}")

    # 2. Total Query Errors Counter
    lines.append("# HELP coderag_query_errors_total Total number of failed RAG queries.")
    lines.append("# TYPE coderag_query_errors_total counter")
    lines.append(f"coderag_query_errors_total {summary.error_count}")

    # 3. Token Consumption Counters
    lines.append("# HELP coderag_tokens_total Total number of LLM tokens consumed by type.")
    lines.append("# TYPE coderag_tokens_total counter")
    lines.append(f'coderag_tokens_total{{type="prompt"}} {summary.total_prompt_tokens}')
    lines.append(f'coderag_tokens_total{{type="completion"}} {summary.total_completion_tokens}')
    lines.append(f'coderag_tokens_total{{type="total"}} {summary.total_tokens}')

    # 4. Total Cost Counter (USD)
    lines.append("# HELP coderag_cost_usd_total Total estimated LLM cost in USD.")
    lines.append("# TYPE coderag_cost_usd_total counter")
    lines.append(f"coderag_cost_usd_total {summary.total_cost_usd:.6f}")

    # 5. Process Uptime Gauge
    lines.append("# HELP coderag_uptime_seconds Total application process uptime in seconds.")
    lines.append("# TYPE coderag_uptime_seconds gauge")
    lines.append(f"coderag_uptime_seconds {summary.uptime_seconds:.1f}")

    # 6. Latency Percentiles (Summary)
    lines.append("# HELP coderag_query_latency_ms Latency percentiles for completed RAG queries in milliseconds.")
    lines.append("# TYPE coderag_query_latency_ms summary")
    lines.append(f'coderag_query_latency_ms{{quantile="0.5"}} {summary.p50_latency_ms:.2f}')
    lines.append(f'coderag_query_latency_ms{{quantile="0.95"}} {summary.p95_latency_ms:.2f}')
    lines.append(f'coderag_query_latency_ms{{quantile="0.99"}} {summary.p99_latency_ms:.2f}')

    # 7. Average Total Latency Gauge
    lines.append("# HELP coderag_latency_avg_ms Average end-to-end query latency in milliseconds.")
    lines.append("# TYPE coderag_latency_avg_ms gauge")
    lines.append(f"coderag_latency_avg_ms {summary.avg_total_latency_ms:.2f}")

    # 8. Pipeline Stage Latencies Gauges
    lines.append("# HELP coderag_stage_latency_ms Average pipeline stage duration in milliseconds.")
    lines.append("# TYPE coderag_stage_latency_ms gauge")
    stage_keys = ["retrieval", "assembly_rerank", "generation", "citations"]
    for stage in stage_keys:
        val = summary.avg_stage_latencies_ms.get(stage, 0.0)
        lines.append(f'coderag_stage_latency_ms{{stage="{stage}"}} {val:.2f}')

    # 9. Optional Extra Gauges (e.g. repos count, collections count)
    if extra_gauges:
        for name, value in extra_gauges.items():
            clean_name = name.strip()
            lines.append(f"# HELP {clean_name} System metric gauge.")
            lines.append(f"# TYPE {clean_name} gauge")
            if isinstance(value, float):
                lines.append(f"{clean_name} {value:.4f}")
            else:
                lines.append(f"{clean_name} {value}")

    # Prometheus format requires a trailing newline
    lines.append("")
    return "\n".join(lines)
