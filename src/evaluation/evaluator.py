"""
Core evaluation engine for Codebase Intelligence RAG.
Orchestrates benchmark test execution, retrieval accuracy benchmarking,
citation fidelity verification, and hallucination / abstention scoring.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from src.evaluation.benchmark import get_default_benchmark
from src.evaluation.metrics import (
    compute_citation_metrics,
    compute_retrieval_metrics,
    compute_symbol_coverage,
    evaluate_abstention,
)
from src.evaluation.models import (
    BenchmarkItem,
    EvaluationReport,
    GenerationMetrics,
    ItemEvalResult,
    RetrievalMetrics,
)
from src.llm.base import LLMProvider, get_llm_provider
from src.rag.engine import RAGEngine
from src.storage.base import VectorStore, sanitize_collection_name
from src.storage.chroma_store import get_vector_store

logger = logging.getLogger(__name__)


class CodebaseEvaluator:
    """
    Evaluates RAG retrieval and generation performance against benchmark Q&A suites.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_provider: LLMProvider | None = None,
        rag_engine: RAGEngine | None = None,
        target_repo: str = "miguelgrinberg/microblog",
        collection_name: str | None = None,
    ):
        if rag_engine is not None:
            self.rag_engine = rag_engine
            self.vector_store = rag_engine.vector_store
            self.llm_provider = rag_engine.llm_provider
        else:
            self.vector_store = vector_store or get_vector_store()
            self.llm_provider = llm_provider or get_llm_provider()
            self.rag_engine = RAGEngine(
                vector_store=self.vector_store,
                llm_provider=self.llm_provider,
                default_collection=collection_name or sanitize_collection_name(target_repo.replace("/", "_")),
            )

        self.target_repo = target_repo
        self.collection_name = collection_name or sanitize_collection_name(target_repo.replace("/", "_"))

    def evaluate_item(
        self,
        item: BenchmarkItem,
        top_k: int = 10,
        evaluate_generation: bool = True,
    ) -> ItemEvalResult:
        """
        Evaluate retrieval and optionally generation for a single benchmark item.
        """
        # Step 1: Evaluate semantic retrieval
        t0 = time.perf_counter()
        retrieved_chunks = self.vector_store.query(
            query_text=item.question,
            top_k=top_k,
            collection_name=self.collection_name,
        )
        retrieval_duration_ms = (time.perf_counter() - t0) * 1000.0

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_chunks=retrieved_chunks,
            expected_files=item.expected_files,
            k=top_k,
        )

        generation_metrics: GenerationMetrics | None = None
        generation_duration_ms = 0.0
        answer_text = ""
        citations_list: list[str] = []

        if evaluate_generation:
            t_gen_0 = time.perf_counter()
            try:
                response = self.rag_engine.answer_question(
                    question=item.question,
                    collection_name=self.collection_name,
                    top_k=top_k,
                )
                generation_duration_ms = (time.perf_counter() - t_gen_0) * 1000.0

                answer_text = response.answer
                citations_list = [c.format_citation() for c in response.citations]

                # Compute citation groundedness & recall
                cit_prec, cit_rec, hallucinated_cits, groundedness = compute_citation_metrics(
                    citations=response.citations,
                    retrieved_sources=response.sources,
                    expected_files=item.expected_files,
                )

                # Compute symbol coverage
                coverage, found_syms, missing_syms = compute_symbol_coverage(
                    answer=answer_text,
                    expected_symbols=item.expected_symbols,
                )

                abstained: bool | None = None
                hallucinated_feat: bool | None = None

                if item.is_negative:
                    abstained, hallucinated_feat = evaluate_abstention(answer_text)

                generation_metrics = GenerationMetrics(
                    citation_precision=cit_prec,
                    citation_recall=cit_rec,
                    hallucinated_citations=hallucinated_cits,
                    groundedness_score=groundedness,
                    symbol_coverage=coverage,
                    symbols_found=found_syms,
                    symbols_missing=missing_syms,
                    abstained_correctly=abstained,
                    hallucinated_feature=hallucinated_feat,
                )
            except Exception as e:
                generation_duration_ms = (time.perf_counter() - t_gen_0) * 1000.0
                logger.warning(f"Generation evaluation failed for item {item.id}: {e}")
                answer_text = f"[Generation Error: {e}]"
                generation_metrics = GenerationMetrics(
                    citation_precision=0.0,
                    citation_recall=0.0,
                    hallucinated_citations=[],
                    groundedness_score=0.0,
                    symbol_coverage=0.0,
                    symbols_found=[],
                    symbols_missing=item.expected_symbols,
                    abstained_correctly=False if item.is_negative else None,
                    hallucinated_feature=True if item.is_negative else None,
                )

        total_latency_ms = retrieval_duration_ms + generation_duration_ms

        # Determine item pass/fail status
        if item.is_negative:
            if generation_metrics and generation_metrics.abstained_correctly is not None:
                passed = generation_metrics.abstained_correctly
            else:
                passed = True
        else:
            passed = retrieval_metrics.hit
            if generation_metrics is not None and generation_metrics.groundedness_score < 0.7:
                passed = False

        preview = (answer_text[:120] + "...") if len(answer_text) > 120 else answer_text

        return ItemEvalResult(
            item_id=item.id,
            question=item.question,
            category=item.category,
            is_negative=item.is_negative,
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            retrieval_latency_ms=round(retrieval_duration_ms, 2),
            generation_latency_ms=round(generation_duration_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            answer_preview=preview,
            citations=citations_list,
            passed=passed,
        )

    def evaluate_benchmark(
        self,
        benchmark_items: list[BenchmarkItem] | None = None,
        top_k: int = 10,
        evaluate_generation: bool = True,
    ) -> EvaluationReport:
        """
        Run the complete evaluation suite across all benchmark items.
        Computes macro-averaged metrics, category breakdowns, and composite quality score.
        """
        items = benchmark_items if benchmark_items is not None else get_default_benchmark()
        results: list[ItemEvalResult] = []

        logger.info(f"Starting evaluation of {len(items)} items for repo '{self.target_repo}' (top_k={top_k})")

        for idx, item in enumerate(items, start=1):
            logger.info(f"[{idx}/{len(items)}] Evaluating {item.id} ('{item.category}')")
            res = self.evaluate_item(item, top_k=top_k, evaluate_generation=evaluate_generation)
            results.append(res)

        # Aggregate standard positive items (non-negative)
        positive_results = [r for r in results if not r.is_negative]
        negative_results = [r for r in results if r.is_negative]

        mean_hit_rate = (
            sum(r.retrieval.hit_rate for r in positive_results) / len(positive_results)
            if positive_results
            else 1.0
        )
        mean_mrr = (
            sum(r.retrieval.mrr for r in positive_results) / len(positive_results)
            if positive_results
            else 1.0
        )
        mean_precision_at_k = (
            sum(r.retrieval.precision_at_k for r in positive_results) / len(positive_results)
            if positive_results
            else 0.0
        )
        mean_recall_at_k = (
            sum(r.retrieval.recall_at_k for r in positive_results) / len(positive_results)
            if positive_results
            else 1.0
        )

        # Aggregate generation metrics
        if evaluate_generation and positive_results:
            gen_pos = [r.generation for r in positive_results if r.generation is not None]
            mean_citation_precision = (
                sum(g.citation_precision for g in gen_pos) / len(gen_pos) if gen_pos else 1.0
            )
            mean_groundedness = (
                sum(g.groundedness_score for g in gen_pos) / len(gen_pos) if gen_pos else 1.0
            )
            mean_symbol_coverage = (
                sum(g.symbol_coverage for g in gen_pos) / len(gen_pos) if gen_pos else 1.0
            )
        else:
            mean_citation_precision = 1.0
            mean_groundedness = 1.0
            mean_symbol_coverage = 1.0

        # Negative abstention and hallucination evaluation
        if negative_results and evaluate_generation:
            correct_abstentions = sum(
                1 for r in negative_results if r.generation and r.generation.abstained_correctly
            )
            abstention_accuracy = correct_abstentions / len(negative_results)
            hallucination_rate = 1.0 - abstention_accuracy
        else:
            abstention_accuracy = 1.0
            hallucination_rate = 0.0

        mean_latency = sum(r.total_latency_ms for r in results) / len(results) if results else 0.0

        # Composite overall score (0.0 to 1.0)
        overall_score = (
            0.30 * mean_hit_rate
            + 0.20 * mean_mrr
            + 0.15 * mean_recall_at_k
            + 0.15 * mean_groundedness
            + 0.10 * mean_symbol_coverage
            + 0.10 * abstention_accuracy
        )

        # Category breakdown
        category_map: dict[str, list[ItemEvalResult]] = defaultdict(list)
        for r in results:
            category_map[r.category].append(r)

        category_breakdown: dict[str, dict[str, float]] = {}
        for cat, cat_res in category_map.items():
            cat_pos = [r for r in cat_res if not r.is_negative]
            if cat_pos:
                hr = sum(r.retrieval.hit_rate for r in cat_pos) / len(cat_pos)
                mrr_val = sum(r.retrieval.mrr for r in cat_pos) / len(cat_pos)
                prec = sum(r.retrieval.precision_at_k for r in cat_pos) / len(cat_pos)
            else:
                hr = 1.0
                mrr_val = 1.0
                prec = 0.0

            g_val = 1.0
            if evaluate_generation:
                cat_gens = [r.generation.groundedness_score for r in cat_res if r.generation]
                if cat_gens:
                    g_val = sum(cat_gens) / len(cat_gens)

            category_breakdown[cat] = {
                "count": len(cat_res),
                "hit_rate": round(hr, 4),
                "mrr": round(mrr_val, 4),
                "precision": round(prec, 4),
                "groundedness": round(g_val, 4),
            }

        provider_name = getattr(self.llm_provider, "model", getattr(self.llm_provider, "model_name", "unknown"))

        return EvaluationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            target_repo=self.target_repo,
            collection_name=self.collection_name,
            total_queries=len(results),
            retrieval_k=top_k,
            llm_provider=provider_name,
            mean_hit_rate=round(mean_hit_rate, 4),
            mean_mrr=round(mean_mrr, 4),
            mean_precision_at_k=round(mean_precision_at_k, 4),
            mean_recall_at_k=round(mean_recall_at_k, 4),
            mean_citation_precision=round(mean_citation_precision, 4),
            mean_groundedness=round(mean_groundedness, 4),
            mean_symbol_coverage=round(mean_symbol_coverage, 4),
            abstention_accuracy=round(abstention_accuracy, 4),
            hallucination_rate=round(hallucination_rate, 4),
            overall_score=round(overall_score, 4),
            mean_latency_ms=round(mean_latency, 2),
            category_breakdown=category_breakdown,
            results=results,
        )
