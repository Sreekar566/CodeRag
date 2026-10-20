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

### ⬜ Phase 6 — Prompt engineering / context assembly
Rank/select chunks within token limits

### ⬜ Phase 7 — Multi-repo support + incremental re-indexing

### ⬜ Phase 8 — Backend API layer (likely FastAPI)

### ⬜ Phase 9 — Frontend/chat UI (likely Streamlit or React)

### ⬜ Phase 10 — Evaluation framework
Retrieval accuracy, hallucination rate against benchmark Q&A pairs

### ⬜ Phase 11 — Testing, CI/CD, containerization, deployment

### ⬜ Phase 12 — Observability, cost monitoring, docs/demo polish

## Tech Stack
- **Confirmed:** Python, GitHub REST API, venv
- **Embedding model:** `flax-sentence-embeddings/st-codesearch-distilroberta-base`
  via sentence-transformers
- **Vector DB:** Chroma (`chromadb==1.5.9`), persistent local storage in `./data/chroma/`
- **Not yet locked in:** frontend framework

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
