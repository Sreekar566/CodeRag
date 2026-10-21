"""
Codebase Intelligence Evaluation Framework.
Provides metrics calculation, benchmark datasets, and automated evaluation runners
for retrieval ranking, groundedness, citation precision, and hallucination / abstention.
"""
from src.evaluation.benchmark import (
    DEFAULT_MICROBLOG_BENCHMARK,
    get_default_benchmark,
    load_benchmark_from_json,
    save_benchmark_to_json,
)
from src.evaluation.evaluator import CodebaseEvaluator
from src.evaluation.metrics import (
    compute_citation_metrics,
    compute_retrieval_metrics,
    compute_symbol_coverage,
    evaluate_abstention,
    match_file_path,
    normalize_file_path,
)
from src.evaluation.models import (
    BenchmarkItem,
    EvaluationReport,
    GenerationMetrics,
    ItemEvalResult,
    RetrievalMetrics,
)

__all__ = [
    "BenchmarkItem",
    "CodebaseEvaluator",
    "DEFAULT_MICROBLOG_BENCHMARK",
    "EvaluationReport",
    "GenerationMetrics",
    "ItemEvalResult",
    "RetrievalMetrics",
    "compute_citation_metrics",
    "compute_retrieval_metrics",
    "compute_symbol_coverage",
    "evaluate_abstention",
    "get_default_benchmark",
    "load_benchmark_from_json",
    "match_file_path",
    "normalize_file_path",
    "save_benchmark_to_json",
]
