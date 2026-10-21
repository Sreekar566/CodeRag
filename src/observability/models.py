"""
Observability and telemetry data models for tracking latency, token usage, API costs,
and ingestion file-filtering audit trails.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any
import uuid


class StageTimer:
    """
    Context manager and stopwatch for precision latency timing (in milliseconds)
    using monotonic performance counters.
    """

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.elapsed_ms: float = 0.0

    def start(self) -> StageTimer:
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def stop(self) -> float:
        if self.start_time is None:
            self.elapsed_ms = 0.0
            return 0.0
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        return self.elapsed_ms

    def __enter__(self) -> StageTimer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


@dataclass
class PricingConfig:
    """
    Model pricing rates per 1,000,000 tokens (USD).
    """
    prompt_rate_per_m: float
    completion_rate_per_m: float


# Default token pricing catalog (USD per 1,000,000 tokens)
MODEL_PRICING: dict[str, PricingConfig] = {
    "fake": PricingConfig(prompt_rate_per_m=0.0, completion_rate_per_m=0.0),
    "fake-model": PricingConfig(prompt_rate_per_m=0.0, completion_rate_per_m=0.0),
    "gemini-3.8-flash": PricingConfig(prompt_rate_per_m=0.15, completion_rate_per_m=0.60),
    "gemini-2.5-flash": PricingConfig(prompt_rate_per_m=0.075, completion_rate_per_m=0.30),
    "gemini-1.5-flash": PricingConfig(prompt_rate_per_m=0.075, completion_rate_per_m=0.30),
    "gemini-1.5-pro": PricingConfig(prompt_rate_per_m=1.25, completion_rate_per_m=5.00),
}

DEFAULT_FALLBACK_PRICING = PricingConfig(prompt_rate_per_m=0.15, completion_rate_per_m=0.60)


def calculate_llm_cost(
    model_name: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float:
    """
    Calculate the estimated cost in USD for an LLM generation based on token counts.
    Returns cost rounded to 7 decimal places.
    """
    if not prompt_tokens and not completion_tokens:
        return 0.0

    p_tokens = prompt_tokens or 0
    c_tokens = completion_tokens or 0

    clean_model = (model_name or "").lower().strip()
    pricing = MODEL_PRICING.get(clean_model)
    if pricing is None:
        # Match by prefix or default fallback
        for key, conf in MODEL_PRICING.items():
            if key in clean_model:
                pricing = conf
                break
        if pricing is None:
            pricing = DEFAULT_FALLBACK_PRICING

    prompt_cost = (p_tokens / 1_000_000.0) * pricing.prompt_rate_per_m
    completion_cost = (c_tokens / 1_000_000.0) * pricing.completion_rate_per_m
    total_cost = prompt_cost + completion_cost
    return round(total_cost, 7)


@dataclass
class QueryTelemetry:
    """
    Telemetry event capturing timing, token usage, and cost for a single RAG query.
    """
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    question: str = ""
    target_repos: list[str] = field(default_factory=list)
    model: str = "unknown"
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    context_tokens: int | None = None
    cost_usd: float = 0.0
    retrieved_count: int = 0
    selected_count: int = 0
    citations_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "target_repos": self.target_repos,
            "model": self.model,
            "stage_latencies_ms": self.stage_latencies_ms,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "context_tokens": self.context_tokens,
            "cost_usd": self.cost_usd,
            "retrieved_count": self.retrieved_count,
            "selected_count": self.selected_count,
            "citations_count": self.citations_count,
            "error": self.error,
        }


@dataclass
class FilteredFileRecord:
    """
    Individual file record rejected during repository ingestion filtering.
    """
    path: str
    reason: str  # e.g., "noise_directory", "noise_extension", "file_too_large"
    rule_match: str  # e.g., "migrations", ".png", "size 600000 > 500000"
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "rule_match": self.rule_match,
            "size_bytes": self.size_bytes,
        }


@dataclass
class IngestionAuditLog:
    """
    Transparency audit log showing every file evaluated during repository ingestion
    and why it was kept or filtered out.
    """
    repo_id: str
    branch: str = "main"
    commit_sha: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_scanned: int = 0
    total_kept: int = 0
    total_filtered: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    kept_files: list[str] = field(default_factory=list)
    filtered_files: list[FilteredFileRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "timestamp": self.timestamp,
            "total_scanned": self.total_scanned,
            "total_kept": self.total_kept,
            "total_filtered": self.total_filtered,
            "reason_counts": self.reason_counts,
            "kept_files": self.kept_files,
            "filtered_files": [f.to_dict() for f in self.filtered_files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionAuditLog:
        filtered = [
            FilteredFileRecord(
                path=item.get("path", ""),
                reason=item.get("reason", "unknown"),
                rule_match=item.get("rule_match", ""),
                size_bytes=item.get("size_bytes"),
            )
            for item in data.get("filtered_files", [])
        ]
        return cls(
            repo_id=data.get("repo_id", "unknown"),
            branch=data.get("branch", "main"),
            commit_sha=data.get("commit_sha"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            total_scanned=data.get("total_scanned", 0),
            total_kept=data.get("total_kept", 0),
            total_filtered=data.get("total_filtered", 0),
            reason_counts=data.get("reason_counts", {}),
            kept_files=data.get("kept_files", []),
            filtered_files=filtered,
        )

    def to_markdown_summary(self) -> str:
        """Format an audit summary table in GitHub Flavored Markdown."""
        lines = [
            f"### Ingestion Transparency Audit: `{self.repo_id}`",
            f"- **Branch:** `{self.branch}`",
            f"- **Commit SHA:** `{self.commit_sha or 'N/A'}`",
            f"- **Evaluated At:** {self.timestamp}",
            f"- **Total Files Scanned:** {self.total_scanned}",
            f"- **Files Kept (Indexed):** {self.total_kept}",
            f"- **Files Filtered Out:** {self.total_filtered}",
            "",
            "#### Filter Reasons Breakdown",
            "| Reason | Category | Count |",
            "| :--- | :--- | :--- |",
        ]
        for reason, count in sorted(self.reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{reason}` | Ingestion Blocklist | {count} |")

        if self.filtered_files:
            lines.extend([
                "",
                "#### Filtered Files Sample (Top 20)",
                "| Path | Reason | Rule Matched | Size |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for f in self.filtered_files[:20]:
                size_str = f"{f.size_bytes:,} B" if f.size_bytes is not None else "N/A"
                lines.append(f"| `{f.path}` | `{f.reason}` | `{f.rule_match}` | {size_str} |")

        return "\n".join(lines)


@dataclass
class MetricsSummary:
    """
    Aggregated system performance, token consumption, and cost metrics summary.
    """
    total_queries: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_total_latency_ms: float = 0.0
    avg_stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_total_latency_ms": round(self.avg_total_latency_ms, 2),
            "avg_stage_latencies_ms": {
                k: round(v, 2) for k, v in self.avg_stage_latencies_ms.items()
            },
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "error_count": self.error_count,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }
