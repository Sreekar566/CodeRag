# Project Log — AI Codebase Intelligence Platform

## Phase 0: Prerequisites and Mental Model — Complete
- Set up Python venv, installed requests, generated GitHub token
- Test repo chosen: miguelgrinberg/microblog
- Core concept: LLMs can't ingest whole repos due to context limits, cost/latency, and "lost in the middle" quality loss - retrieval (RAG) solves this

## Phase 1: Basic GitHub Repo Ingestion — Complete
Built src/ingestion/github_client.py:
- get_repo_tree(owner, repo, token, branch="main") - recursive file tree via GitHub API
- get_file_content(owner, repo, path, token) - fetches + base64-decodes file content
Verified: 73 files found in microblog, real source fetched from app/auth/routes.py
Lessons: tree API vs contents API differ; never hardcode file paths; 404 can mean bad assumption not bad code; rate limits are a shared resource across concurrent users

## Phase 2: File Filtering and Code Chunking — Complete
Resolved: filter decision reframed as "does this file help answer a target question type" not "is this code vs config."
- .flaskenv, LICENSE, .gitattributes -> filter out (settings/legal/git-internal, no behavior, not deployment-relevant)
- Procfile, Vagrantfile -> keep (onboarding/deployment questions are in scope for "summarize repo for a new engineer")
Filter design: blocklist strategy chosen over allowlist (generalizes across languages without enumerating every source extension; accepted tradeoff is fail-open risk vs allowlist's fail-closed). Three layers: (1) noise directory blocklist, (2) noise extension/filename blocklist, (3) file size threshold as a red flag for generated/data files. Extensionless files (Dockerfile, Makefile, Procfile) pass through automatically under blocklist - no special-casing needed.
Considered and rejected: human-in-the-loop review of ambiguous/extensionless binary files during ingestion. Rejected because (1) it breaks the automated-pipeline promise and is incompatible with async background workers (Phase 7), and (2) it's redundant - get_file_content() from Phase 1 already hard-fails to decode true binaries, so there's no real ambiguity left for a human to resolve.
Implemented `filter_files(tree)` in `src/ingestion/filters.py` and `chunk_text(text, path)` in `src/ingestion/chunker.py` with sliding window and blank-line boundary snapping.

## Phase 3: Embedding Layer & Test Infrastructure — Complete
What was built:
- Unit test suite: `tests/test_filters.py`, `tests/test_chunker.py`, `tests/test_github_client.py` (mocked GitHub API), and `tests/test_embedder.py` (mocked model).
- Pinned `pytest==9.1.1`, `iniconfig==2.3.1`, and `pluggy==1.7.0` in `requirements.txt`.
- Configured `pytest.ini` with `integration` marker to skip tests requiring network or real model by default (`addopts = -m "not integration"`).
- Language metadata detection: added `detect_language(path)` in `src/ingestion/chunker.py` covering standard languages and filenames (Dockerfile, Makefile, Procfile, Vagrantfile), ensuring chunks include `language` metadata.
- Lazy model loading: refactored `src/embedding/embedder.py` to use `get_model()` and dynamic module property `__getattr__("model")` instead of loading weights at import time.
- Batched embedding in `embed_chunks(chunks)` with `.tolist()` serialization and fallback language tagging.
- Verified end-to-end: unit tests pass in ~4.4s, and integration test runs real model generating 768-dimensional embeddings on CPU.

Design options considered:
1. Model initialization timing:
   - Option A: Load SentenceTransformer at module import time.
   - Option B: Lazy loading with caching on first `get_model()` / `embed_chunks()` call.
   - Choice made: Option B.
   - Why: Loading at import time forced tests to load ~328MB model weights and allocate >1GB RAM on every test session, and would fail in network-isolated environments. Lazy loading keeps imports fast (~milliseconds) and respects the testing rule that unit tests must run without network or model downloads.
2. Language detection mechanism:
   - Option A: External library (e.g. Pygments, python-magic).
   - Option B: Lightweight extension and filename mapping dictionary.
   - Choice made: Option B.
   - Why: Avoids unnecessary dependencies and native library bindings, executes in sub-microsecond time, and accurately covers all repository code and configuration files.
3. Language tagging lifecycle:
   - Option A: Tag only during chunking in `chunk_text()`.
   - Option B: Tag during `chunk_text()` AND guarantee presence in `embed_chunks()`.
   - Choice made: Option B.
   - Why: Defense-in-depth: guarantees that any chunk passed to `embed_chunks()` carries language metadata before vector DB insertion in Phase 4.

## Phase 4: Vector DB Storage & Semantic Retrieval — Complete
What was built:
- Abstract interface: `src/storage/base.py` defining `VectorStore` ABC (`add_chunks`, `query`, `query_by_embedding`, `get`, `count`, `clear`, `delete`, `list_collections`) and `sanitize_collection_name()`.
- Chroma vector store: `src/storage/chroma_store.py` (`ChromaVectorStore`) with local persistent storage in `./data/chroma/`, cosine similarity (`hnsw:space: cosine`), batch upserting (batch size 500), metadata sanitization, and structured result formatting. Factory function `get_vector_store()`.
- In-memory vector store: `src/storage/memory_store.py` (`InMemoryVectorStore`) using numpy cosine similarity for zero-dependency test runs and swappable backend validation.
- Ingestion-to-storage pipeline: `src/storage/indexer.py` implementing `index_repository(owner, repo, token, ...)` and `index_files(files, ...)` connecting GitHub fetching, file filtering, chunking, embedding, and vector storage with error handling for skipped/corrupt files.
- Embedding helpers: added `embed_query(query, model)` and `embed_texts(texts, model)` to `src/embedding/embedder.py`.
- Unit tests: `tests/test_storage.py` and `tests/test_indexer.py` covering all CRUD operations, filtering, deterministic ID generation, sanitization, and error handling (35 unit tests pass in ~8s).
- Integration test: `tests/test_storage_integration.py` verifying real persistent Chroma vector indexing and semantic retrieval for natural-language code queries.
- Dependency management: pinned `chromadb==1.5.9` and all transitive dependencies in `requirements.txt`.

Design options considered:
1. Vector Store abstraction:
   - Option A: Direct Chroma API calls embedded in ingestion and retrieval scripts.
   - Option B: Generic `VectorStore` ABC with pluggable implementations (`ChromaVectorStore`, `InMemoryVectorStore`).
   - Choice made: Option B.
   - Why: Decouples vector DB specifics from ingestion and downstream RAG logic, enables fast unit testing without disk or DB overhead, and satisfies the requirement that the vector store must be swappable.
2. Embedding generation for query and store:
   - Option A: Rely on Chroma's built-in default embedding function.
   - Option B: Pass `embedding_function=None` to Chroma and use our code-specialized embedder (`src/embedding/embedder.py`).
   - Choice made: Option B.
   - Why: Chroma's default embedding function tries to download a generic text ONNX model from the internet, violating offline unit test constraints and producing vector dimension mismatches against our 768-dim code embeddings. Disabling Chroma's internal embedder guarantees exact model alignment and zero network access during testing.
3. Chunk ID strategy:
   - Option A: Random UUIDs for each chunk.
   - Option B: Deterministic, human-readable IDs formatted as `{path}:{start_line}-{end_line}` with index disambiguation.
   - Choice made: Option B.
   - Why: Deterministic IDs make `add_chunks` idempotent (re-indexing a repo or updating chunks overwrites the same keys rather than duplicating vectors), and make retrieved chunk IDs immediately interpretable for citation.

## Deferred to later phases
- Caching repeated ingestion (Phase 6/8)
- Concurrency handling for many users (Phase 7)
- Auto-detect default branch instead of hardcoding "main"
- Async/parallel file fetching (optimization, post-MVP)
- Post-ingestion transparency/audit log showing what was filtered and why, for user inspection (pairs naturally with Phase 11 observability, not a blocking ingestion-time prompt)
