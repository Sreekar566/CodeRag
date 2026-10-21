"""
Data models for the Codebase Intelligence Evaluation Framework.
Defines benchmark schemas, retrieval metrics, groundedness & citation metrics,
and comprehensive benchmark evaluation reports.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class BenchmarkItem:
    """
    A single benchmark question with ground truth annotations.
    """
    id: str
    question: str
    expected_files: list[str] = field(default_factory=list)
    expected_symbols: list[str] = field(default_factory=list)
    ground_truth: str = ""
    category: str = "general"
    is_negative: bool = False
    target_repo: str = "miguelgrinberg/microblog"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkItem:
        return cls(
            id=data["id"],
            question=data["question"],
            expected_files=list(data.get("expected_files", [])),
            expected_symbols=list(data.get("expected_symbols", [])),
            ground_truth=data.get("ground_truth", ""),
            category=data.get("category", "general"),
            is_negative=bool(data.get("is_negative", False)),
            target_repo=data.get("target_repo", "miguelgrinberg/microblog"),
        )


@dataclass
class RetrievalMetrics:
    """
    Quantitative evaluation metrics for semantic retrieval.
    """
    hit: bool
    hit_rate: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    retrieved_files: list[str] = field(default_factory=list)
    relevant_files_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationMetrics:
    """
    Quantitative evaluation metrics for LLM generation, citation accuracy, and groundedness.
    """
    citation_precision: float
    citation_recall: float
    hallucinated_citations: list[str] = field(default_factory=list)
    groundedness_score: float = 1.0
    symbol_coverage: float = 1.0
    symbols_found: list[str] = field(default_factory=list)
    symbols_missing: list[str] = field(default_factory=list)
    abstained_correctly: bool | None = None
    hallucinated_feature: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ItemEvalResult:
    """
    Evaluation outcome for an individual benchmark item.
    """
    item_id: str
    question: str
    category: str
    is_negative: bool
    retrieval: RetrievalMetrics
    generation: GenerationMetrics | None = None
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    answer_preview: str = ""
    citations: list[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        return res


@dataclass
class EvaluationReport:
    """
    Aggregated benchmark evaluation report across all test cases.
    """
    timestamp: str
    target_repo: str
    collection_name: str
    total_queries: int
    retrieval_k: int
    llm_provider: str
    mean_hit_rate: float
    mean_mrr: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_citation_precision: float
    mean_groundedness: float
    mean_symbol_coverage: float
    abstention_accuracy: float
    hallucination_rate: float
    overall_score: float
    mean_latency_ms: float
    category_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[ItemEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "target_repo": self.target_repo,
            "collection_name": self.collection_name,
            "total_queries": self.total_queries,
            "retrieval_k": self.retrieval_k,
            "llm_provider": self.llm_provider,
            "summary_metrics": {
                "mean_hit_rate": round(self.mean_hit_rate, 4),
                "mean_mrr": round(self.mean_mrr, 4),
                "mean_precision_at_k": round(self.mean_precision_at_k, 4),
                "mean_recall_at_k": round(self.mean_recall_at_k, 4),
                "mean_citation_precision": round(self.mean_citation_precision, 4),
                "mean_groundedness": round(self.mean_groundedness, 4),
                "mean_symbol_coverage": round(self.mean_symbol_coverage, 4),
                "abstention_accuracy": round(self.abstention_accuracy, 4),
                "hallucination_rate": round(self.hallucination_rate, 4),
                "overall_score": round(self.overall_score, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 2),
            },
            "category_breakdown": self.category_breakdown,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_summary_table(self) -> str:
        """Format high-level metrics into a concise terminal-ready ASCII table."""
        lines = [
            "============================================================",
            f" RAG BENCHMARK EVALUATION REPORT — {self.target_repo}",
            f" Provider: {self.llm_provider} | Top-K: {self.retrieval_k} | Queries: {self.total_queries}",
            "============================================================",
            f" Retrieval Hit Rate @ {self.retrieval_k}:  {self.mean_hit_rate * 100:6.1f}%",
            f" Mean Reciprocal Rank (MRR):      {self.mean_mrr:6.3f}",
            f" Retrieval Precision @ {self.retrieval_k}: {self.mean_precision_at_k * 100:6.1f}%",
            f" Retrieval Recall @ {self.retrieval_k}:    {self.mean_recall_at_k * 100:6.1f}%",
            "------------------------------------------------------------",
            f" Citation Precision:              {self.mean_citation_precision * 100:6.1f}%",
            f" Citation Groundedness:           {self.mean_groundedness * 100:6.1f}%",
            f" Symbol / Keyword Coverage:       {self.mean_symbol_coverage * 100:6.1f}%",
            f" Abstention Accuracy (Negatives): {self.abstention_accuracy * 100:6.1f}%",
            f" Hallucination Rate:              {self.hallucination_rate * 100:6.1f}%",
            "------------------------------------------------------------",
            f" Overall Quality Score:           {self.overall_score * 100:6.1f}%",
            f" Average Latency:                 {self.mean_latency_ms:6.1f} ms",
            "============================================================",
        ]
        return "\n".join(lines)

    def format_markdown_report(self) -> str:
        """Format evaluation findings into a GitHub-flavored markdown report."""
        table = [
            f"# RAG Benchmark Evaluation: `{self.target_repo}`",
            "",
            f"**Timestamp:** {self.timestamp}  ",
            f"**LLM Provider:** `{self.llm_provider}`  ",
            f"**Retrieval Depth (Top-K):** `{self.retrieval_k}`  ",
            f"**Total Benchmark Items:** `{self.total_queries}`  ",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Score | Description |",
            "| :--- | :--- | :--- |",
            f"| **Retrieval Hit Rate @ {self.retrieval_k}** | `{self.mean_hit_rate * 100:.1f}%` | Percentage of queries where at least one target file was retrieved |",
            f"| **Mean Reciprocal Rank (MRR)** | `{self.mean_mrr:.3f}` | Reciprocal rank of the first relevant file in retrieved chunks |",
            f"| **Retrieval Precision @ {self.retrieval_k}** | `{self.mean_precision_at_k * 100:.1f}%` | Fraction of retrieved chunks belonging to target files |",
            f"| **Retrieval Recall @ {self.retrieval_k}** | `{self.mean_recall_at_k * 100:.1f}%` | Fraction of expected target files retrieved in top-k |",
            f"| **Citation Precision** | `{self.mean_citation_precision * 100:.1f}%` | Fraction of cited files that are relevant target files |",
            f"| **Citation Groundedness** | `{self.mean_groundedness * 100:.1f}%` | Citations supported by retrieved context (non-hallucinated) |",
            f"| **Symbol / Keyword Coverage** | `{self.mean_symbol_coverage * 100:.1f}%` | Core codebase functions/classes present in answer |",
            f"| **Abstention Accuracy** | `{self.abstention_accuracy * 100:.1f}%` | Accuracy in rejecting out-of-scope / negative probe queries |",
            f"| **Hallucination Rate** | `{self.hallucination_rate * 100:.1f}%` | Frequency of invented files, citations, or non-existent features |",
            f"| **Overall Quality Score** | `{self.overall_score * 100:.1f}%` | Composite weighted index of retrieval, groundedness, and accuracy |",
            f"| **Average Latency** | `{self.mean_latency_ms:.1f} ms` | Mean round-trip latency per query |",
            "",
            "## Category Performance",
            "",
            "| Category | Queries | Hit Rate | MRR | Precision | Groundedness |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for cat, stats in sorted(self.category_breakdown.items()):
            count = int(stats.get("count", 0))
            hr = stats.get("hit_rate", 0.0) * 100
            mrr = stats.get("mrr", 0.0)
            p = stats.get("precision", 0.0) * 100
            g = stats.get("groundedness", 1.0) * 100
            table.append(f"| `{cat}` | {count} | {hr:.1f}% | {mrr:.3f} | {p:.1f}% | {g:.1f}% |")

        table.extend([
            "",
            "## Itemized Evaluation Results",
            "",
            "| ID | Category | Question | Hit? | MRR | Citations Grounded? | Latency | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for r in self.results:
            hit_str = "✅" if r.retrieval.hit or r.is_negative else "❌"
            mrr_str = f"{r.retrieval.mrr:.2f}"
            ground_str = "N/A"
            if r.generation is not None:
                ground_str = "✅" if r.generation.groundedness_score >= 0.99 else f"{r.generation.groundedness_score * 100:.0f}%"
            status_str = "✅ PASS" if r.passed else "⚠️ REVIEW"
            table.append(
                f"| `{r.item_id}` | `{r.category}` | {r.question} | {hit_str} | {mrr_str} | {ground_str} | {r.total_latency_ms:.0f}ms | {status_str} |"
            )

        return "\n".join(table)
