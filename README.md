# AI Codebase Intelligence Platform

A production-grade, multi-repository code intelligence and question-answering system powered by Retrieval-Augmented Generation (RAG). It ingests public GitHub repositories, builds semantic vector indexes over code structures, and provides grounded answers citing exact file paths and line ranges.

---

## Features

- **Language-Agnostic Ingestion & Blocklist Filtering:** Recursively fetches repository file trees via GitHub API without downloading bulky git history; filters binaries, lockfiles, build artifacts, test caches, and noise directories.
- **Boundary-Snapping Chunking:** Splits code using an overlapping sliding window (`CHUNK_SIZE=400`, `OVERLAP=50`) that snaps to nearest blank lines to preserve cohesive functions without requiring fragile per-language AST parsers.
- **Semantic Code Embeddings:** Converts code chunks into 768-dimensional dense vectors using `flax-sentence-embeddings/st-codesearch-distilroberta-base`.
- **Persistent Vector Storage:** Backed by local Chroma vector database (`./data/chroma/`) with cosine similarity metrics and swappable storage interface.
- **Incremental Git Sync:** Uses Git Merkle tree blob SHAs to compute file diffs (`added`, `modified`, `deleted`, `unchanged`). Only modified and added files are re-chunked and embedded, saving rate limits and computation.
- **Multi-Signal Re-Ranking & Context Budgeting:** Reranks chunks combining vector similarity, lexical keywords, file path relevance, and symbol definitions (`def`, `class`). Assembles context within strict token budgets using "lost-in-the-middle" alternating placement.
- **Grounded LLM Q&A:** Pluggable LLM interface supporting Google Gemini (`gemini-3.8-flash`) and deterministic `fake` provider for offline testing. Cites exact file spans (`path/to/file.py:10-25`).
- **Pipeline Observability & Latency Profiling:** High-precision monotonic stage timing breaking down query execution across retrieval, context assembly/reranking, LLM generation, and citation extraction.
- **LLM Token & Cost Monitoring:** Automated token usage accounting and per-query USD cost tracking with model-specific pricing catalogs and percentile metrics (p50, p95, p99).
- **Ingestion Transparency & Audit Logs:** Complete audit trail tracking every file evaluated during ingestion, logging why each file was kept or filtered out (noise directory, noise extension, oversized file).
- **REST API (FastAPI):** Comprehensive endpoints for repository management, indexing, background tasks, diff preview, collections, multi-repo queries, system metrics, and audit logs.
- **Interactive Web UI (Streamlit):** Web interface for chat, grounded citation review, syntax-highlighted chunk inspection, incremental repo indexing, Git diff checks, and observability dashboard.

---

## Architecture Overview

```
GitHub API ──> Filter (Blocklist) ──> Blank-line Snapping Chunker
                                               │
                                               ▼
Chroma Vector DB <── st-codesearch Embedding Layer
       │
       ▼
Semantic Retrieval ──> Multi-Signal Reranker ──> Token Budget Assembler
                                                         │
                                                         ▼
                                                Gemini LLM Engine
                                                         │
                                                         ▼
                                       Grounded Answer + Line Citations
```

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- Linux / macOS (4 GB RAM minimum; runs on CPU)

### 1. Clone & Set Up Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch (CPU-only) & Dependencies

```bash
# Install PyTorch CPU wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file or export environment variables:

```bash
# GitHub Access Token (for accessing public repositories without low rate limits)
export GITHUB_TOKEN="ghp_your_personal_access_token"

# LLM Provider: 'gemini' for real generation, 'fake' for offline/testing
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your_gemini_api_key"

# Optional UI settings
export API_BASE_URL="http://localhost:8000"
```

---

## Quickstart & Usage

### Option 1: Interactive Streamlit Web UI

Start both the backend server and frontend interface:

1. **Start the FastAPI Backend:**
   ```bash
   python -m src.api.main --host 0.0.0.0 --port 8000
   ```

2. **Start the Streamlit Frontend UI:**
   ```bash
   python -m src.ui.main --port 8501
   ```
   Open `http://localhost:8501` in your browser.

In the Web UI:
- **💬 Codebase Chat:** Ask natural language questions, inspect grounded citations (`path/to/file.py:10-25`), view syntax-highlighted source chunks, and view latency breakdown and estimated query cost.
- **📦 Repositories & Ingestion:** Ingest any GitHub repository (e.g. `miguelgrinberg/microblog`), check pending Git diffs, view file filtering audit logs, or execute incremental re-indexing.
- **🔍 Vector Collections:** Browse stored vector collections, sample chunks, and inspect code metadata.
- **📊 Observability & Costs:** Real-time metrics dashboard tracking total queries, tokens consumed, estimated API cost in USD, pipeline stage latency distributions (retrieval, assembly, generation, citations), recent query logs, and repository file filter audit inspector.

---

### Option 2: REST API

With the backend running (`python -m src.api.main`):

- **Interactive API Docs:** Visit `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc`.
- **System Health:**
  ```bash
  curl http://localhost:8000/health
  ```
- **Index a Repository:**
  ```bash
  curl -X POST http://localhost:8000/index \
       -H "Content-Type: application/json" \
       -d '{"owner": "miguelgrinberg", "repo": "microblog", "branch": "main"}'
  ```
- **Ask a Question:**
  ```bash
  curl -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -d '{"question": "How is user authentication handled?", "repo": "miguelgrinberg/microblog"}'
  ```
- **System Metrics & Cost Monitoring:**
  ```bash
  curl http://localhost:8000/metrics
  curl http://localhost:8000/metrics/queries?limit=10
  ```
- **Repository Ingestion Audit Log:**
  ```bash
  curl http://localhost:8000/repos/miguelgrinberg/microblog/audit
  ```

---

### Option 3: Python CLI / Programmatic Usage

You can also interact directly via Python:

```python
from src.registry.incremental import IncrementalIndexer
from src.rag.engine import ask_codebase

# 1. Ingest & index repository
indexer = IncrementalIndexer()
result = indexer.index_repository_incremental(
    owner="miguelgrinberg",
    repo="microblog",
)
print(f"Indexed {result.files_processed} files into {result.chunks_stored} chunks.")

# 2. Ask a question
response = ask_codebase(
    question="Where is database configuration set up?",
    repo="miguelgrinberg/microblog",
)

print(response.formatted_answer)
for citation in response.citations:
    print(f"  - {citation.citation_str}")
```

---

### Option 4: Docker & Docker Compose

The platform is fully containerized with CPU-only PyTorch and multi-service topology:

1. **Start Full Stack with Docker Compose:**
   ```bash
   # Set your environment variables (or define in .env)
   export GITHUB_TOKEN="ghp_your_token"
   export GEMINI_API_KEY="your_gemini_key"
   export LLM_PROVIDER="gemini"

   # Build and start FastAPI backend + Streamlit UI
   docker compose up --build
   ```

   - **Backend API:** Available at `http://localhost:8000` (Swagger UI at `/docs`)
   - **Frontend UI:** Available at `http://localhost:8501`
   - **Persistent Volume:** Vector database and repo registry persist in the `codebase_data` volume.

2. **Run Standalone Docker Container:**
   ```bash
   # Build image
   docker build -t ai-codebase-intel:latest .

   # Launch FastAPI backend
   docker run -d --name codebase-api -p 8000:8000 \
     -v $(pwd)/data:/app/data \
     -e GITHUB_TOKEN="ghp_your_token" \
     -e GEMINI_API_KEY="your_gemini_key" \
     ai-codebase-intel:latest api

   # Launch Streamlit UI connected to backend
   docker run -d --name codebase-ui -p 8501:8501 \
     -e API_BASE_URL="http://localhost:8000" \
     ai-codebase-intel:latest ui
   ```

---

## Continuous Integration & CI/CD

A production-ready GitHub Actions CI pipeline is configured in `.github/workflows/ci.yml`:
- **Automated Testing:** Sets up Python 3.12, installs CPU PyTorch and pinned dependencies, and executes the complete offline test suite (`pytest -v`).
- **Container Validation:** Verifies `docker-compose.yml` configuration and validates Docker image builds using Buildx.
- **Branch Triggers:** Executes on every `push` and `pull_request` targeting `main` and `master`.

---

The platform includes an automated evaluation framework to measure retrieval accuracy (Hit Rate@K, MRR, Precision@K, Recall@K), citation grounding fidelity, and hallucination / abstention rates against benchmark questions:

```bash
# Run benchmark evaluation on miguelgrinberg/microblog
python -m src.evaluation.main --repo miguelgrinberg/microblog

# Run fast retrieval-only ranking evaluation
python -m src.evaluation.main --repo miguelgrinberg/microblog --retrieval-only --top-k 10

# Export Markdown and JSON evaluation reports
python -m src.evaluation.main --output data/eval_report.json --markdown data/eval_report.md
```

### Metrics Tracked
- **Hit Rate @ K:** Frequency with which target ground truth files appear in top-K retrieved chunks.
- **Mean Reciprocal Rank (MRR):** Rank position of the first relevant chunk in search results.
- **Citation Groundedness:** Proportion of cited file spans that actually exist within retrieved context chunks (measuring zero hallucinated citations).
- **Abstention Accuracy & Hallucination Rate:** Evaluates how reliably the system declines to answer unanswerable or out-of-scope negative probes (e.g. asking for Stripe or Kubernetes in a Flask microblog).

---

## Running Tests

All unit tests run offline without network access, model downloads, or API keys:

```bash
# Run the complete offline test suite
pytest

# Run tests with verbose output
pytest -v

# Run integration tests (requires network, PyTorch model, and GEMINI_API_KEY)
pytest -m integration
```