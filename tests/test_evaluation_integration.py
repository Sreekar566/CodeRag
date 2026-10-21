"""
Integration tests for the Codebase Intelligence Evaluation Framework.
Marked with @pytest.mark.integration and skipped by default.
"""
from __future__ import annotations

import os
import pytest

from src.evaluation.benchmark import get_default_benchmark
from src.evaluation.evaluator import CodebaseEvaluator
from src.llm.base import get_llm_provider
from src.storage.chroma_store import get_vector_store


@pytest.mark.integration
def test_evaluation_live_retrieval_microblog():
    """Verify live retrieval evaluation against real indexed microblog in Chroma."""
    vector_store = get_vector_store()
    llm_provider = get_llm_provider()

    evaluator = CodebaseEvaluator(
        vector_store=vector_store,
        llm_provider=llm_provider,
        target_repo="miguelgrinberg/microblog",
        collection_name="miguelgrinberg_microblog",
    )

    items = get_default_benchmark()[:3]
    report = evaluator.evaluate_benchmark(benchmark_items=items, top_k=5, evaluate_generation=False)

    assert report.total_queries == 3
    assert report.retrieval_k == 5
    assert report.mean_hit_rate > 0.0
    assert len(report.results) == 3
