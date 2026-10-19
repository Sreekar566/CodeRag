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

### ⬜ Phase 3 — Embedding layer
Turn code chunks into vectors + metadata (file path, line numbers,
language)

### ⬜ Phase 4 — Vector DB storage & semantic retrieval

### ⬜ Phase 5 — RAG / Q&A engine
Retrieve relevant chunks, feed to LLM, generate cited answers

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
- **Not yet locked in:** embedding model, vector DB (candidates:
  Chroma, FAISS, Pinecone), frontend framework

## Test Repo
`miguelgrinberg/microblog` — used for all Phase 0–2 development/testing

## Working Conventions
- Ingestion code lives under `src/ingestion/`
- Follow existing patterns in `github_client.py` for new modules
  (function signatures take `owner, repo, token` style params)
- Flag design tradeoffs before implementing — this is a learning/
  resume project, so reasoning matters as much as working code
- Don't skip ahead to later phases without confirming current
  phase is done
