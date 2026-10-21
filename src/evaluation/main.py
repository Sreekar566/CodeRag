"""
CLI runner for the Codebase Intelligence Evaluation Framework.
Allows executing benchmark evaluation runs against any indexed repository
and generating JSON and Markdown reports.

Usage:
    python -m src.evaluation.main --repo miguelgrinberg/microblog
    python -m src.evaluation.main --retrieval-only --top-k 5
    python -m src.evaluation.main --output data/eval_report.json --markdown data/eval_report.md
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.evaluation.benchmark import get_default_benchmark, load_benchmark_from_json
from src.evaluation.evaluator import CodebaseEvaluator
from src.llm.base import get_llm_provider
from src.storage.base import sanitize_collection_name
from src.storage.chroma_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation suite against indexed codebase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="miguelgrinberg/microblog",
        help="Target GitHub repository (owner/repo)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Vector store collection name (defaults to sanitized owner_repo)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to custom benchmark questions JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval rankings, skipping LLM generation",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        choices=["fake", "gemini"],
        help="LLM provider for generation evaluation (overrides LLM_PROVIDER env)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to write JSON evaluation report",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="File path to write Markdown evaluation report",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.llm:
        os.environ["LLM_PROVIDER"] = args.llm

    target_repo = args.repo
    col_name = args.collection or sanitize_collection_name(target_repo.replace("/", "_"))

    if args.benchmark:
        benchmark_items = load_benchmark_from_json(args.benchmark)
    else:
        benchmark_items = get_default_benchmark()

    vector_store = get_vector_store()
    llm_provider = get_llm_provider()

    evaluator = CodebaseEvaluator(
        vector_store=vector_store,
        llm_provider=llm_provider,
        target_repo=target_repo,
        collection_name=col_name,
    )

    evaluate_generation = not args.retrieval_only

    report = evaluator.evaluate_benchmark(
        benchmark_items=benchmark_items,
        top_k=args.top_k,
        evaluate_generation=evaluate_generation,
    )

    # Print summary table to console
    print("\n" + report.format_summary_table() + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.to_json(indent=2))
        logger.info(f"JSON evaluation report saved to {out_path}")

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.format_markdown_report())
        logger.info(f"Markdown evaluation report saved to {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
