# AI Codebase Intelligence Platform

## Project Overview
Resume/portfolio project: an AI system that ingests any public GitHub
repository and answers natural-language questions about its code via
RAG (retrieval-augmented generation) — grounded, cited answers, not
hallucinated summaries.

## Roadmap
Phased build, Phase 0 through Phase 12. Do not assume a phase is
complete unless explicitly confirmed below.

## Status

### ✅ Phase 0 — Setup (complete)
- Python venv configured
- GitHub personal access token working

### ✅ Phase 1 — Ingestion (complete)
- File: `src/ingestion/github_client.py`
- `get_repo_tree(owner, repo, token, branch="main")` — recursively
  lists all files in a repo via GitHub API
- `get_file_content(owner, repo, path, token)` — fetches +
  base64-decodes a single file's content
- Tested against `miguelgrinberg/microblog` — 73 files found,
  confirmed real content pulled from `app/auth/routes.py`

### ✅ Phase 2 — File Filtering & Code Chunking (complete)
- File inclusion strategy: **blocklist**, not allowlist — chosen because
  it generalizes across languages without enumerating every source
  extension; accepted tradeoff is fail-open risk vs. an allowlist's
  fail-closed guarantee
- Three-layer filter in `src/ingestion/filters.py`: (1) noise directory
  blocklist (`NOISE_DIRS`), (2) noise extension/filename blocklist
  (`NOISE_EXTENSIONS`), (3) file size threshold (`MAX_FILE_SIZE`) as a
  red flag for generated/data files
- Decision framing: "does this file help answer a target question type,"
  not "is this code vs. config"
- Excluded: git metadata (`.gitattributes`, `.gitignore`, `.git/`), env
  config (`.flaskenv`), legal (`LICENSE`), build/dependency directories
  (`node_modules`, `__pycache__`, `venv`/`.venv`, `.pytest_cache`,
  `dist`, `build`), binary/media/archive extensions, `migrations/`,
  `.po` translation files
- Kept in scope: deploy/onboarding tooling (`Procfile`, `Vagrantfile`,
  `Dockerfile`) — onboarding/deployment questions are in scope for
  "summarize repo for a new engineer"
- Verified against all 54 filtered files in `miguelgrinberg/microblog`:
  no crashes, no unfiltered noise
- Chunking strategy: **sliding window** (token-based, approximated via
  whitespace-split word count until an embedding model/tokenizer is
  chosen), with overlap — chosen over per-function/AST-aware chunking
  because it stays language-agnostic and avoids a per-language parser;
  accepted tradeoff is chunks can still land mid-function
- Two safeguards added to reduce that risk without full AST parsing:
  (1) chunk boundaries snap to the nearest blank line within
  `BOUNDARY_SEARCH_WINDOW` lines rather than cutting at an exact word
  count, (2) each chunk is tagged with file path + line range so later
  phases can expand context around a retrieved chunk
- Built `src/ingestion/chunker.py`: `chunk_text(text, path)` —
  `CHUNK_SIZE=400`, `OVERLAP=50`, `BOUNDARY_SEARCH_WINDOW=5`
- Verified end-to-end against all 54 filtered microblog files: no
  crashes, no gaps in line coverage between chunks

### ✅ Phase 3 — Embedding layer (complete)
Turn code chunks into vectors + metadata (file path, line numbers,
language)
- `src/embedding/embedder.py` written: `embed_chunks(chunks)` using
  `flax-sentence-embeddings/st-codesearch-distilroberta-base`
- Added language detection (`detect_language` in `chunker.py`) and ensured
  language metadata is included on all chunks
- Refactored model loading to be lazy (`get_model()`) to avoid network/memory
  overhead during imports and unit tests
- Verified end-to-end with unit tests (mocked model) and integration tests
  producing 768-dimensional embeddings

### ✅ Phase 4 — Vector DB storage & semantic retrieval (complete)
- Created `src/storage/` package with swappable `VectorStore` interface (`base.py`)
- Implemented `ChromaVectorStore` (`chroma_store.py`) backed by Chroma with persistent storage in `./data/chroma/` (cosine similarity metric `hnsw:space: cosine`)
- Implemented `InMemoryVectorStore` (`memory_store.py`) as an alternative swappable backend
- Implemented idempotent chunk ID generation, metadata type sanitization, and batch upserting
- Implemented semantic retrieval with `query` (text search via embedder) and `query_by_embedding`, returning ranked results with chunk metadata and line ranges
- Added `index_repository` and `index_files` in `src/storage/indexer.py` connecting ingestion, filtering, chunking, embedding, and vector storage
- Pinned `chromadb==1.5.9` and its dependencies in `requirements.txt`
- Verified end-to-end with unit tests (35 passing) and integration test confirming semantic search accuracy against real embeddings

### ✅ Phase 5 — RAG / Q&A engine (complete)
- Created `src/llm/` package with swappable `LLMProvider` interface (`base.py`) selected via `LLM_PROVIDER` env var (`fake`, `gemini`)
- Built `FakeLLMProvider` (`fake.py`) for deterministic, offline testing with support for canned answers, callback functions, invocation history, and prompt context extraction
- Built `GeminiLLMProvider` (`gemini.py`) using official `google-genai` SDK and `GEMINI_API_KEY`
- Pinned `google-genai==2.31.0` and dependencies in `requirements.txt`
- Created `src/rag/` package with `RAGEngine` (`engine.py`), structured `RAGResponse`, and `Citation` data models
- Implemented grounded prompt assembly (`prompt.py`) formatting source chunks with file paths, line ranges, and language fences
- Implemented citation extraction and fallback reconciliation (`citations.py`) parsing regex code line citations and filtering false positives (URLs)
- Added high-level convenience function `ask_codebase`
- Added unit test suites `tests/test_llm.py` and `tests/test_rag.py` (61 unit tests passing in ~8s)
- Added live integration test `tests/test_rag_integration.py` verifying real Gemini generation and citations

### ✅ Phase 6 — Prompt engineering / context assembly (complete)
- Created token utilities in `src/rag/tokenizer.py` (`count_tokens`, `truncate_text_to_tokens`, `estimate_chunk_tokens`) using `tiktoken` (cl100k_base) with robust regex/heuristic fallbacks
- Created multi-signal re-ranking engine in `src/rag/reranker.py` (`ReRanker`, `rerank_chunks`) combining semantic similarity with lexical keyword matches, file path relevance, and function/class definition detection
- Created adjacent/overlapping chunk deduplication and merging in `src/rag/reranker.py` (`merge_overlapping_chunks`)
- Created context budget assembler in `src/rag/context_assembler.py` (`ContextAssembler`, `assemble_context`) enforcing strict token budgets, line-aware chunk truncation, and attention-optimal "lost in the middle" alternating reordering
- Enhanced prompt formatting in `src/rag/prompt.py` (`build_rag_prompt`, `format_code_context`) supporting both raw chunks and pre-budgeted context strings with truncation indicators
- Integrated context assembler into `RAGEngine` (`src/rag/engine.py`) and `ask_codebase` with metadata tracking (`retrieved_count`, `selected_count`, `context_tokens`, `truncated_chunks_count`, `dropped_chunks_count`, `reranked`)
- Pinned `tiktoken==0.14.0` in `requirements.txt`
- Added comprehensive unit test suites `tests/test_tokenizer.py`, `tests/test_reranker.py`, and `tests/test_context_assembler.py` (87 unit tests passing in ~8.6s)

### ✅ Phase 7 — Multi-repo support + incremental re-indexing (complete)
- Created `src/registry/` package with `RepositoryMetadata`, `ChangeSet`, `IncrementalIndexResult`, `RepoRegistry`, and `IncrementalIndexer`
- Added default branch auto-detection (`get_default_branch`) and latest commit SHA tracking (`get_latest_commit_sha`) in `src/ingestion/github_client.py`
- Implemented persistent JSON repository registry (`./data/registry.json`) with atomic file writes and per-repo re-entrant locks for concurrency safety
- Implemented file tree SHA diffing in `detect_changes()` categorizing files into added, modified, deleted, and unchanged
- Implemented incremental synchronization with vector store: unchanged files are never re-fetched or re-embedded, modified/deleted files have old chunks purged, and newly added/modified chunks are embedded and indexed
- Implemented multi-repo querying and cross-codebase RAG retrieval in `RAGEngine` (`answer_question` and `answer_multi_repo`), ranking chunks across repositories within context token budget with repo-annotated citations
- Added unit and integration tests across `test_registry.py`, `test_incremental.py`, and `test_multi_repo.py` (112 unit tests passing in ~9.1s)

### ✅ Phase 8 — Backend API layer (complete)
- Created `src/api/` package with FastAPI application (`app.py`, `create_app`) and CLI runner (`main.py`)
- Pydantic v2 request/response schemas (`src/api/models.py`) with strict validation:
  - `IndexRequest`, `QueryRequest`, `HealthResponse`, `RepoResponse`, `RepoListResponse`, `IndexResponse`, `QueryResponse`, `CitationModel`, `SourceChunkModel`, `CollectionResponse`, `CollectionListResponse`, `CollectionDetailResponse`, `ChangeSetResponse`, `DeleteResponse`, `ErrorResponse`
- Modular routing architecture (`src/api/routes/`):
  - `GET /health` — health check reporting vector store type, active LLM provider, repo & collection counts
  - `GET /repos` & `GET /repos/{owner}/{repo}` — list indexed repositories and retrieve metadata
  - `DELETE /repos/{owner}/{repo}` — remove repository from registry and purge collection from vector store
  - `GET /repos/{owner}/{repo}/changes` — preview pending GitHub tree changes without modifying vector store
  - `POST /index` — synchronous or background (`background=true`, HTTP 202) repository indexing with lock conflict handling (HTTP 409)
  - `POST /query` — codebase natural-language Q&A across single or multiple repositories with citations and token budgeting
  - `GET /collections`, `GET /collections/{name}`, `DELETE /collections/{name}` — vector collection inspection and management
- Dependency injection architecture (`src/api/dependencies.py`) supporting seamless `app.dependency_overrides` for offline testing
- CORSMiddleware configured for cross-origin frontend support
- Pinned `fastapi==0.156.3` and `starlette==1.9.0` in `requirements.txt`
- Comprehensive unit test suite in `tests/test_api.py` (20 tests passing in ~5.9s; total 132 tests passing)

### ✅ Phase 9 — Frontend/chat UI (complete)
- Created `src/ui/` package with interactive Streamlit web dashboard and CLI runner (`python -m src.ui.main`)
- `ApiClient` (`src/ui/client.py`) connecting to backend FastAPI server with connection diagnostics, timeout handling, and domain error classes
- Reusable UI presentation components (`src/ui/components.py`):
  - Code citation rendering with file path and line numbers
  - Syntax-highlighted code chunk viewer with similarity scores and line range metadata
  - Remote Git Merkle tree diff previewing pending repository changes
  - Formatted query metrics (tokens, retrieved/selected chunks, reranked status)
- Streamlit application (`src/ui/app.py`):
  - 💬 **Codebase Chat**: interactive Q&A scoped to single or all repositories, retrieval controls (top-k, token budget, reranking, temperature), suggested question pills, grounded markdown generation, citations panel, and source chunk inspection
  - 📦 **Repositories & Ingestion**: GitHub repository indexing with branch auto-detection, force re-indexing, background tasks, repository explorer, remote Git diff checks, one-click incremental sync, and cascading deletion
  - 🔍 **Vector Collections**: vector store collection browser, sample chunk inspection, and collection purging
  - ℹ️ **Architecture & About**: RAG pipeline explanation and links to FastAPI Swagger UI docs
  - Responsive Sidebar: backend connectivity indicator, vector store & LLM model telemetry, and connection settings
- CLI launcher (`src/ui/main.py`) accepting `--host`, `--port`, and `--api-url` arguments
- Root `README.md` created with complete setup, CLI, REST API, and Streamlit UI documentation
- Pinned `streamlit==1.66.0` and its dependencies in `requirements.txt`
- Unit test suite `tests/test_ui.py` (20 tests passing in ~0.3s; total 152 tests passing)

### ✅ Phase 10 — Evaluation framework (complete)
- Created `src/evaluation/` package:
  - `models.py`: Data models for `BenchmarkItem`, `RetrievalMetrics`, `GenerationMetrics`, `ItemEvalResult`, and `EvaluationReport` with ASCII summary tables, GitHub Flavored Markdown reports, and JSON export.
  - `metrics.py`: Deterministic metrics computation: Hit Rate@K, Mean Reciprocal Rank (MRR), Precision@K, Recall@K, Citation Groundedness (verifying citations exist in retrieved context), Symbol/Keyword Coverage, and Abstention / Hallucination Rate on negative probes.
  - `benchmark.py`: Curated benchmark suite for `miguelgrinberg/microblog` covering authentication, database models, password reset email, application factory, configuration, error handling, route pagination, and adversarial negative probes (`microblog-neg-stripe`, `microblog-neg-k8s`). Includes JSON loader and dataset persistence.
  - `evaluator.py`: `CodebaseEvaluator` orchestrating automated benchmark evaluations across vector stores (Chroma, InMemory) and LLM providers (Fake, Gemini), with resilient error handling and category performance breakdowns.
  - `main.py`: Command-line evaluation runner (`python -m src.evaluation.main`) supporting `--repo`, `--collection`, `--benchmark`, `--top-k`, `--retrieval-only`, `--llm`, `--output`, and `--markdown`.
- Evaluated end-to-end against real indexed `miguelgrinberg/microblog` repository:
  - Retrieval Hit Rate @ 10: 100.0%
  - Mean Reciprocal Rank (MRR): 0.423
  - Retrieval Recall @ 10: 85.0%
  - Citation Groundedness: 100.0%
  - Negative Probe Abstention Accuracy: 100.0%
  - Hallucination Rate: 0.0%
  - Overall Quality Score: 76.2%
  - Average Latency: 334.1 ms
- Comprehensive unit test suite `tests/test_evaluation.py` (17 tests passing in ~8s offline; total 169 unit tests passing) and integration test `tests/test_evaluation_integration.py`.

### ✅ Phase 11 — Testing, CI/CD, containerization, deployment (complete)
- Created production-ready `Dockerfile`:
  - Based on `python:3.12-slim` with `PYTHONUNBUFFERED=1` and minimal footprint
  - Non-root security user `appuser:appgroup` (UID/GID 1000)
  - Installs CPU-only PyTorch wheel first (`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`)
  - Installs pinned dependencies from `requirements.txt`
  - Multi-service entrypoint supporting `api`, `ui`, `eval`, and `test`
  - Exposes port 8000 (FastAPI backend) and 8501 (Streamlit frontend)
- Created `docker-compose.yml`:
  - Orchestrates `api` and `ui` services with healthchecks (`curl -f http://localhost:8000/health`)
  - Dependency ordering with `condition: service_healthy`
  - Named persistent volume `codebase_data` for vector database and registry
  - Configurable environment variables for `PORT`, `API_BASE_URL`, `LLM_PROVIDER`, `GEMINI_API_KEY`, `GITHUB_TOKEN`
- Created `.dockerignore` excluding `.git`, `venv`, `data/`, `.pytest_cache`, `__pycache__`, and secrets
- Created GitHub Actions CI workflow `.github/workflows/ci.yml`:
  - Triggers on push and pull requests to `main` and `master`
  - Python 3.12 environment with pip caching
  - Installs CPU PyTorch and requirements
  - Runs full offline test suite (`pytest -v`)
  - Validates `docker compose config` and Docker image build
- Enhanced `CHROMA_PERSIST_DIRECTORY` and `REGISTRY_PATH` environment variable overrides across storage and registry layers
- Comprehensive unit test suite `tests/test_deployment.py` (6 unit tests, total 175 tests passing)
- Updated documentation in `README.md` with Docker and Docker Compose instructions

### ⬜ Phase 12 — Observability, cost monitoring, docs/demo polish

## Tech Stack
- **Confirmed:** Python, GitHub REST API, venv
- **Embedding model:** `flax-sentence-embeddings/st-codesearch-distilroberta-base`
  via sentence-transformers
- **Vector DB:** Chroma (`chromadb==1.5.9`), persistent local storage in `./data/chroma/`
- **Frontend framework:** Streamlit (`streamlit==1.66.0`)

## Test Repo
`miguelgrinberg/microblog` — used for all Phase 0–2 development/testing

## Working Conventions
- Ingestion code lives under `src/ingestion/`, embedding code under
  `src/embedding/`; each new phase gets its own package under `src/`
- Follow existing patterns in `github_client.py` for new modules
  (function signatures take `owner, repo, token` style params)
- Record design tradeoffs in `project_log.md` (options considered, choice,
  reasoning) as part of implementing each phase
- Work through phases in order; a phase is only marked ✅ once its code
  runs end-to-end and its tests pass
