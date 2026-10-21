"""
Unit tests for the Codebase Intelligence Evaluation Framework.
Tests metrics calculation, benchmark loading, offline evaluation execution,
hallucination / abstention detection, and report generation.
Runs completely offline with zero network and mock/in-memory components.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.evaluation.benchmark import (
    DEFAULT_MICROBLOG_BENCHMARK,
    get_default_benchmark,
    load_benchmark_from_json,
    save_benchmark_to_json,
)
from src.evaluation.evaluator import CodebaseEvaluator
from src.evaluation.main import parse_args, main as cli_main
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
from src.llm.fake import FakeLLMProvider
from src.rag.citations import Citation
from src.storage.memory_store import InMemoryVectorStore


def test_normalize_file_path():
    assert normalize_file_path("app/auth/routes.py") == "app/auth/routes.py"
    assert normalize_file_path("/app/auth/routes.py") == "app/auth/routes.py"
    assert normalize_file_path("./app/auth/routes.py") == "app/auth/routes.py"
    assert normalize_file_path("app\\auth\\routes.py") == "app/auth/routes.py"
    assert normalize_file_path("   app/auth/routes.py   ") == "app/auth/routes.py"
    assert normalize_file_path("") == ""


def test_match_file_path():
    assert match_file_path("app/auth/routes.py", "app/auth/routes.py") is True
    assert match_file_path("repo/app/auth/routes.py", "app/auth/routes.py") is True
    assert match_file_path("app/auth/routes.py", "repo/app/auth/routes.py") is True
    assert match_file_path("app/models.py", "app/auth/routes.py") is False
    assert match_file_path("", "app/models.py") is False


def test_retrieval_metrics_exact_hit_first_rank():
    retrieved = [
        {"path": "app/auth/routes.py"},
        {"path": "app/models.py"},
        {"path": "config.py"},
    ]
    expected = ["app/auth/routes.py"]
    metrics = compute_retrieval_metrics(retrieved, expected, k=3)

    assert metrics.hit is True
    assert metrics.hit_rate == 1.0
    assert metrics.mrr == 1.0
    assert metrics.precision_at_k == round(1 / 3, 4)
    assert metrics.recall_at_k == 1.0
    assert metrics.relevant_files_found == ["app/auth/routes.py"]


def test_retrieval_metrics_hit_second_rank():
    retrieved = [
        {"path": "config.py"},
        {"path": "app/auth/routes.py"},
        {"path": "app/models.py"},
    ]
    expected = ["app/auth/routes.py"]
    metrics = compute_retrieval_metrics(retrieved, expected, k=3)

    assert metrics.hit is True
    assert metrics.hit_rate == 1.0
    assert metrics.mrr == 0.5  # 1 / 2
    assert metrics.recall_at_k == 1.0


def test_retrieval_metrics_miss():
    retrieved = [
        {"path": "config.py"},
        {"path": "app/models.py"},
    ]
    expected = ["app/auth/routes.py"]
    metrics = compute_retrieval_metrics(retrieved, expected, k=5)

    assert metrics.hit is False
    assert metrics.hit_rate == 0.0
    assert metrics.mrr == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.relevant_files_found == []


def test_retrieval_metrics_negative_probe():
    retrieved = [
        {"path": "config.py"},
        {"path": "app/models.py"},
    ]
    metrics = compute_retrieval_metrics(retrieved, expected_files=[], k=5)

    assert metrics.hit is True
    assert metrics.hit_rate == 1.0
    assert metrics.mrr == 1.0
    assert metrics.relevant_files_found == []


def test_citation_metrics_fully_grounded():
    citations = [
        Citation(path="app/auth/routes.py", start_line=10, end_line=25),
        Citation(path="app/auth/forms.py", start_line=5, end_line=15),
    ]
    sources = [
        {"path": "app/auth/routes.py", "start_line": 1, "end_line": 50},
        {"path": "app/auth/forms.py", "start_line": 1, "end_line": 20},
    ]
    expected = ["app/auth/routes.py", "app/auth/forms.py"]

    prec, rec, hall, groundedness = compute_citation_metrics(citations, sources, expected)
    assert prec == 1.0
    assert rec == 1.0
    assert hall == []
    assert groundedness == 1.0


def test_citation_metrics_hallucinated():
    citations = [
        Citation(path="app/auth/routes.py", start_line=10, end_line=25),
        Citation(path="app/invented/file.py", start_line=1, end_line=5),  # Not in sources!
    ]
    sources = [
        {"path": "app/auth/routes.py", "start_line": 1, "end_line": 50},
    ]
    expected = ["app/auth/routes.py"]

    prec, rec, hall, groundedness = compute_citation_metrics(citations, sources, expected)
    assert groundedness == 0.5  # 1 out of 2 grounded
    assert "app/invented/file.py" in hall
    assert prec == 0.5
    assert rec == 1.0


def test_symbol_coverage():
    answer = "The User class defines set_password and check_password methods along with avatar."
    expected = ["User", "set_password", "check_password", "non_existent_method"]

    coverage, found, missing = compute_symbol_coverage(answer, expected)
    assert coverage == 0.75
    assert "User" in found
    assert "set_password" in found
    assert "check_password" in found
    assert "non_existent_method" in missing


def test_symbol_coverage_empty():
    assert compute_symbol_coverage("Some text", []) == (1.0, [], [])
    assert compute_symbol_coverage("", ["User"]) == (0.0, [], ["User"])


def test_evaluate_abstention():
    answer_abstained = "Stripe payment processing is not implemented in this codebase."
    abstained, hall = evaluate_abstention(answer_abstained)
    assert abstained is True
    assert hall is False

    answer_hallucinated = "Stripe payments are handled by the StripeWebhookHandler in app/billing/routes.py."
    abstained, hall = evaluate_abstention(answer_hallucinated)
    assert abstained is False
    assert hall is True


def test_benchmark_item_model():
    item = BenchmarkItem(
        id="test-1",
        question="How does login work?",
        expected_files=["app/auth/routes.py"],
        expected_symbols=["login"],
        ground_truth="Login view validates LoginForm",
        category="auth",
    )
    d = item.to_dict()
    assert d["id"] == "test-1"
    assert d["expected_files"] == ["app/auth/routes.py"]

    restored = BenchmarkItem.from_dict(d)
    assert restored.id == item.id
    assert restored.question == item.question
    assert restored.expected_files == item.expected_files


def test_benchmark_save_and_load(tmp_path: Path):
    items = get_default_benchmark()
    assert len(items) >= 10

    test_file = tmp_path / "test_benchmark.json"
    save_benchmark_to_json(test_file, items)
    assert test_file.exists()

    loaded = load_benchmark_from_json(test_file)
    assert len(loaded) == len(items)
    assert loaded[0].id == items[0].id


def test_evaluator_end_to_end_offline():
    # Setup offline in-memory vector store with sample chunks
    vector_store = InMemoryVectorStore()
    col = "test_eval_col"

    chunks = [
        {
            "text": "def login():\n    form = LoginForm()\n    if form.validate_on_submit():\n        return redirect(url_for('main.index'))",
            "path": "app/auth/routes.py",
            "start_line": 10,
            "end_line": 20,
            "language": "python",
            "embedding": [0.2] * 768,
        },
        {
            "text": "class User(db.Model):\n    id = db.Column(db.Integer, primary_key=True)\n    username = db.Column(db.String(64))",
            "path": "app/models.py",
            "start_line": 5,
            "end_line": 15,
            "language": "python",
            "embedding": [0.1] * 768,
        },
    ]
    vector_store.add_chunks(chunks, collection_name=col)

    # Use query embedding hook so query returns chunks
    vector_store._embed_fn = lambda text: [0.2] * 768

    fake_llm = FakeLLMProvider(
        canned_response="User login is handled in app/auth/routes.py:10-20 using LoginForm and redirect."
    )

    evaluator = CodebaseEvaluator(
        vector_store=vector_store,
        llm_provider=fake_llm,
        target_repo="test/repo",
        collection_name=col,
    )

    items = [
        BenchmarkItem(
            id="q1",
            question="How does login work?",
            expected_files=["app/auth/routes.py"],
            expected_symbols=["login", "LoginForm"],
            category="authentication",
        ),
        BenchmarkItem(
            id="q2-neg",
            question="Where is Stripe payment processing?",
            expected_files=[],
            expected_symbols=[],
            category="negative_probe",
            is_negative=True,
        ),
    ]

    report = evaluator.evaluate_benchmark(benchmark_items=items, top_k=2, evaluate_generation=True)

    assert report.total_queries == 2
    assert report.retrieval_k == 2
    assert report.mean_hit_rate == 1.0
    assert report.mean_mrr == 1.0
    assert report.mean_groundedness >= 0.0
    assert len(report.results) == 2
    assert "authentication" in report.category_breakdown

    # Test report formatting
    summary_str = report.format_summary_table()
    assert "RAG BENCHMARK EVALUATION REPORT" in summary_str

    md_str = report.format_markdown_report()
    assert "# RAG Benchmark Evaluation" in md_str
    assert "| `q1` |" in md_str

    json_str = report.to_json()
    assert "mean_hit_rate" in json_str


def test_evaluator_retrieval_only():
    vector_store = InMemoryVectorStore()
    col = "test_retrieval_col"
    chunk = {
        "text": "class Config:\n    SECRET_KEY = 'test'",
        "path": "config.py",
        "start_line": 1,
        "end_line": 5,
        "language": "python",
        "embedding": [0.3] * 768,
    }
    vector_store.add_chunks([chunk], collection_name=col)
    vector_store._embed_fn = lambda text: [0.3] * 768

    evaluator = CodebaseEvaluator(
        vector_store=vector_store,
        llm_provider=FakeLLMProvider(),
        target_repo="test/repo",
        collection_name=col,
    )

    item = BenchmarkItem(
        id="cfg-1",
        question="Where is configuration stored?",
        expected_files=["config.py"],
    )

    res = evaluator.evaluate_item(item, top_k=1, evaluate_generation=False)
    assert res.retrieval.hit is True
    assert res.retrieval.mrr == 1.0
    assert res.generation is None
    assert res.passed is True


def test_cli_parsing():
    args = parse_args(["--repo", "foo/bar", "--top-k", "5", "--retrieval-only"])
    assert args.repo == "foo/bar"
    assert args.top_k == 5
    assert args.retrieval_only is True


def test_cli_execution_with_fake_llm(tmp_path: Path):
    # Test CLI execution with output files
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    # Minimal benchmark
    bench_file = tmp_path / "mini_bench.json"
    save_benchmark_to_json(
        bench_file,
        [
            BenchmarkItem(
                id="mini-1",
                question="What is the app factory?",
                expected_files=["app/__init__.py"],
            )
        ],
    )

    exit_code = cli_main(
        [
            "--repo",
            "dummy/repo",
            "--benchmark",
            str(bench_file),
            "--top-k",
            "2",
            "--retrieval-only",
            "--output",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
    )

    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()
    with open(out_json) as f:
        data = json.load(f)
        assert data["total_queries"] == 1
