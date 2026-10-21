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

## Phase 5: RAG / Q&A Engine & LLM Interface — Complete
What was built:
- LLM Provider Abstraction: `src/llm/base.py` defining `LLMProvider` ABC (`generate`, `complete`) and `LLMResponse` dataclass.
- Factory helper `get_llm_provider()` reading `LLM_PROVIDER` env var, falling back to Gemini if `GEMINI_API_KEY` exists, or `FakeLLMProvider` by default.
- Fake LLM Provider: `src/llm/fake.py` (`FakeLLMProvider`) returning deterministic canned responses, tracking call count and prompt history, and extracting context lines from prompt for realistic grounded testing.
- Gemini Provider: `src/llm/gemini.py` (`GeminiLLMProvider`) implementing Google GenAI integration via official `google-genai` SDK using `GEMINI_API_KEY` and configurable models (defaulting to `gemini-3.8-flash`).
- Citation & Grounding Layer: `src/rag/citations.py` (`Citation` model, `extract_citations_from_text` with regex and URL filtering, and `citations_from_chunks`).
- Prompt Context Assembly: `src/rag/prompt.py` (`DEFAULT_SYSTEM_PROMPT`, `format_code_context`, `build_rag_prompt`) formatting retrieved chunks into clear, fenced markdown blocks.
- RAG Engine: `src/rag/engine.py` (`RAGEngine` and `ask_codebase()`) orchestrating semantic retrieval via `VectorStore.query`, zero-result handling without hallucinations, prompt construction, LLM generation, and citation reconciliation into a structured `RAGResponse`.
- Unit tests: `tests/test_llm.py` (14 unit tests) and `tests/test_rag.py` (12 unit tests), bringing total unit test suite to 61 passing tests in ~8s without network or API keys.
- Integration tests: `tests/test_rag_integration.py` verifying real Gemini generation and RAG retrieval when `GEMINI_API_KEY` is present.
- Pinned `google-genai==2.31.0` and dependencies in `requirements.txt`.

Design options considered:
1. LLM Provider integration strategy:
   - Option A: Couple the RAG engine directly to Google's SDK.
   - Option B: Swappable `LLMProvider` interface with factory function and provider selection via `LLM_PROVIDER`.
   - Choice made: Option B.
   - Why: Core architectural requirement; decouples prompt generation and citation extraction from specific model providers, and allows unit tests to run fully offline with zero token costs and deterministic assertions.
2. Citation extraction mechanism:
   - Option A: Require structured JSON outputs from the LLM.
   - Option B: Instruct LLM to cite in natural text (e.g. `path/to/file.py:10-25`) with regex parsing and fallback to retrieved chunk sources.
   - Choice made: Option B.
   - Why: JSON mode constrains reasoning and causes failures when models produce markdown or conversational context. Natural citation parsing with strict URL filtering handles both in-text citations and guarantees source attribution even when a model summarizes without repeating line numbers.
3. Empty retrieval handling:
   - Option A: Pass empty context to the LLM and let it answer from parametric memory.
   - Option B: Short-circuit when 0 chunks are found, returning an explicit grounded message without calling the LLM.
   - Choice made: Option B.
   - Why: Prevents hallucinations on unindexed or empty repositories, guarantees groundedness, and saves LLM token budget.

## Phase 6: Prompt Engineering / Context Assembly — Complete
What was built:
- Token utilities in `src/rag/tokenizer.py`: `count_tokens`, `truncate_text_to_tokens`, `estimate_chunk_tokens` using `tiktoken` (`cl100k_base` encoding) with LRU caching and regex word/punctuation fallback heuristics.
- Code-aware re-ranking in `src/rag/reranker.py`: `ReRanker` and `rerank_chunks` incorporating query keyword extraction (handling snake_case and camelCase identifiers), lexical overlap scoring, file path relevance scoring, and symbol definition detection (`def`, `class`, `function`, `const`).
- Overlapping & adjacent chunk merging in `src/rag/reranker.py`: `merge_overlapping_chunks` automatically combines sliding-window chunks from the same file into unified snippets, eliminating duplicate context lines and expanding continuous code blocks.
- Token budget & context assembly in `src/rag/context_assembler.py`: `ContextAssembler` and `assemble_context` enforcing configurable token limits (`max_context_tokens`, default 2048), line-preserving partial truncation of borderline chunks, and attention-optimal ("lost in the middle") alternating placement.
- Prompt construction enhancements in `src/rag/prompt.py`: updated `build_rag_prompt` and `format_code_context` to handle pre-budgeted context blocks, numbered source blocks, and truncation warnings.
- Integrated context assembly into `RAGEngine` in `src/rag/engine.py` and `ask_codebase` convenience function, populating metadata (`retrieved_count`, `selected_count`, `context_tokens`, `max_context_tokens`, `truncated_chunks_count`, `dropped_chunks_count`, `reranked`).
- Pinned `tiktoken==0.14.0` in `requirements.txt`.
- Unit test suites: `tests/test_tokenizer.py` (9 tests), `tests/test_reranker.py` (9 tests), `tests/test_context_assembler.py` (6 tests), plus RAG engine integration tests (87 passing unit tests in ~8.6s).

Design options considered:
1. Tokenizer implementation:
   - Option A: Word-count approximation only (`len(text.split()) * 1.3`).
   - Option B: `tiktoken` BPE tokenizer (`cl100k_base`) with graceful fallback to regex/word heuristics if unavailable.
   - Choice made: Option B.
   - Why: BPE tokenization accurately reflects modern LLM context limits (critical when handling code indentation, brackets, and syntax characters that word splitting grossly misestimates), while the heuristic fallback prevents hard crashes in restricted environments.
2. Handling chunks exceeding remaining token budget:
   - Option A: Strict hard drop (discard any chunk that doesn't fit in remaining tokens).
   - Option B: Clean line-preserving truncation for the boundary chunk (if remaining tokens >= threshold), discarding only subsequent chunks.
   - Choice made: Option B.
   - Why: Maximizes token window utilization and allows relevant parts of a high-ranking chunk (e.g., function signature and top lines) into context rather than leaving an empty gap in the budget.
3. Attention decay ("Lost in the Middle"):
   - Option A: Retain naive descending rank order in context.
   - Option B: Reorder selected chunks placing top-ranked snippets at the outermost edges (start and end of context) and lower-ranked snippets in the center.
   - Choice made: Option B.
   - Why: Empirical research confirms LLM retrieval attention is strongest at prompt boundaries (near instructions and user query) and degrades in the middle; alternating placement significantly reduces hallucination and improves answer accuracy.

## Phase 7: Multi-repo Support & Incremental Re-indexing — Complete
What was built:
- Enhanced GitHub client in `src/ingestion/github_client.py`: added `get_default_branch(owner, repo, token)` to query repository metadata and `get_latest_commit_sha(owner, repo, token, branch)` to query latest commit on a branch.
- Dedicated `src/registry/` package:
  - `src/registry/models.py`: structured data models `RepositoryMetadata` (tracking owner, repo, branch, commit SHA, file SHAs mapping, chunk count, operational status), `ChangeSet` (classifying added, modified, deleted, unchanged files), and `IncrementalIndexResult`.
  - `src/registry/registry.py`: `RepoRegistry` with persistent JSON storage (`./data/registry.json`), atomic disk writes (temp file + `os.replace`), thread safety (`threading.RLock`), and per-repository locks to avoid concurrent duplicate indexing jobs.
  - `src/registry/incremental.py`: `detect_changes` computing tree SHA differences, and `IncrementalIndexer` (`index_repository_incremental`) orchestrating delta-only synchronization with VectorStore.
- Ingestion-to-VectorStore synchronization:
  - Commit SHA short-circuiting: unchanged repositories skip tree and file fetching entirely.
  - Modified files: purge stale chunks from vector store before re-indexing.
  - Deleted files: purge chunks from vector store using metadata queries (`where={"path": path}`).
  - Unchanged files: zero network calls to GitHub, zero chunking, zero embedding computation.
- Multi-repository RAG support in `src/rag/engine.py`:
  - `answer_question` supports `repos: list[str]`, `collections: list[str]`, or multi-repo lists, querying each collection and tagging candidate chunks with repo labels.
  - `answer_multi_repo(question, repos)` convenience method for cross-repo questions.
  - Cross-repo context budgeting and re-ranking: best snippets across codebases compete fairly within the token budget.
  - Prompt formatting in `src/rag/prompt.py` annotates sources with `Repo: ...` tags.
  - Citations in `src/rag/citations.py` support repository attribution (`repo:path:start-end`).
- Unit and integration tests:
  - `tests/test_registry.py` (7 tests): serialization, CRUD, disk persistence, concurrency, and locks.
  - `tests/test_incremental.py` (8 tests): diff detection, initial indexing, unchanged skips, delta updates, branch auto-detection, and persistent Chroma integration.
  - `tests/test_multi_repo.py` (4 tests): cross-repo queries, prompt annotations, citations, and zero-token empty handling.
  - Total unit test suite: 112 passed, 5 integration tests deselected by default.

Design options considered:
1. Incremental change detection mechanism:
   - Option A: Fetch all file contents and compute MD5/SHA256 hashes locally.
   - Option B: Use GitHub Git Tree blob SHAs (already provided in the single tree API response) and compare against registry records.
   - Choice made: Option B.
   - Why: Option A requires $O(N)$ HTTP API requests to GitHub on every re-indexing run, rapidly exhausting GitHub rate limits. Option B utilizes git's native Merkle tree blob SHAs in a single tree API call, reducing API calls to $O(M)$ where $M$ is the number of modified/added files only.
2. Vector store cleanup for modified files:
   - Option A: Rely purely on chunk ID overwrites via upsert.
   - Option B: Explicitly delete old chunks for the file path before adding new chunks (`where={"path": path}`).
   - Choice made: Option B.
   - Why: Upsert only overwrites chunks with identical IDs. If a file shrank from 6 chunks to 3 chunks after modification, pure upsert leaves 3 orphaned chunks in the vector DB, polluting subsequent retrieval with phantom code. Explicit path-scoped deletion guarantees complete cleanup.
3. Multi-repo query architecture:
   - Option A: Consolidate all codebases into a single vector collection with a repo metadata tag.
   - Option B: Maintain distinct collections per repository and federate search across collections at query time.
   - Choice made: Option B.
   - Why: Per-repo collections maintain clean isolation, make incremental re-indexing, clearing, and deleting trivial without expensive metadata filtering across massive indices, and prevent HNSW index degradation across hundreds of repos.

## Phase 8: Backend API Layer — Complete
What was built:
- Dedicated `src/api/` package with modular FastAPI application and CLI runner (`python -m src.api.main`):
  - `src/api/models.py`: Pydantic v2 schemas (`IndexRequest`, `QueryRequest`, `HealthResponse`, `RepoResponse`, `RepoListResponse`, `IndexResponse`, `QueryResponse`, `CitationModel`, `SourceChunkModel`, `CollectionResponse`, `CollectionListResponse`, `CollectionDetailResponse`, `ChangeSetResponse`, `DeleteResponse`, `ErrorResponse`).
  - `src/api/dependencies.py`: Dependency injection providers (`get_registry`, `get_storage`, `get_llm`, `get_indexer`, `get_rag_engine`) enabling isolated overrides for test suites.
  - Modular route handlers in `src/api/routes/`:
    - `health.py`: `GET /health` reporting system status, vector store implementation, LLM model, registered repos count, and collection count.
    - `repos.py`: `GET /repos` (list registered codebases), `GET /repos/{owner}/{repo}` (repository details), `DELETE /repos/{owner}/{repo}` (cascading removal from registry and vector store), `GET /repos/{owner}/{repo}/changes` (Git Merkle tree diff inspection without vector store modification).
    - `index.py`: `POST /index` supporting both synchronous and asynchronous background execution (`background=true`, HTTP 202 Accepted) with concurrency lock contention detection (HTTP 409 Conflict).
    - `query.py`: `POST /query` executing token-budgeted RAG retrieval across single or multiple codebases, returning answers, formatted citations, and ranked source chunks.
    - `collections.py`: `GET /collections` (list collections and chunk counts), `GET /collections/{name}` (collection chunk count and sample code snippets), `DELETE /collections/{name}` (purge collection from vector store).
  - `src/api/app.py`: FastAPI factory (`create_app`) with OpenAPI documentation (`/docs`, `/redoc`), root route `/`, and `CORSMiddleware` configured for cross-origin frontend communication.
  - Pinned `fastapi==0.156.3` and `starlette==1.9.0` in `requirements.txt`.
  - Comprehensive unit test suite `tests/test_api.py` (20 tests covering all endpoints, edge cases, error codes, and CORS headers; all passing in ~5.9s offline). Total project unit test suite: 132 tests passing.

Design options considered:
1. Long-running repository indexing API behavior:
   - Option A: Only synchronous indexing blocking the HTTP connection until complete.
   - Option B: Support both synchronous execution and asynchronous background task execution (`background=true`) via FastAPI's `BackgroundTasks`, returning HTTP 202 Accepted.
   - Choice made: Option B.
   - Why: Large repositories with hundreds of files can take tens of seconds to fetch, chunk, and embed, which causes standard HTTP client/proxy timeouts (typically 30-60s) and poor UI responsiveness. Background execution with status tracking in the registry provides resilient operations while preserving synchronous execution for small repos, scripts, and deterministic testing.
2. API Layer dependency management:
   - Option A: Global singletons imported directly inside route functions.
   - Option B: FastAPI `Depends()` dependency injection system (`src/api/dependencies.py`).
   - Choice made: Option B.
   - Why: `Depends()` allows unit and integration tests to cleanly override services via `app.dependency_overrides` (e.g. substituting `InMemoryVectorStore` and `FakeLLMProvider` with zero disk or network overhead), preventing test pollution and ensuring fast execution.
3. Repository deletion semantics:
   - Option A: Only delete repository metadata from the registry, leaving vectors in storage.
   - Option B: Cascade deletion: purge the associated collection from the vector store and remove metadata from the registry.
   - Choice made: Option B.
   - Why: Orphaned vector store collections consume disk space and memory and create discrepancies between the registry and storage. Atomic cascading deletion guarantees consistent state.

## Phase 9: Frontend / Chat UI (Streamlit) — Complete
What was built:
- Dedicated `src/ui/` package with interactive Streamlit web dashboard and CLI runner (`python -m src.ui.main`):
  - `src/ui/client.py`: `ApiClient` wrapping the FastAPI backend with connection retry/timeout handling, custom domain exceptions (`ApiClientError`, `ApiConnectionError`), and methods for system health checks, repository CRUD, Git diff previews, synchronous/background indexing, vector collection inspection, and natural-language RAG queries.
  - `src/ui/components.py`: Presentation formatting and rendering components for code citations (`path:start-end`), execution metrics (tokens, chunk counts, latency), status badges with indicators, Git diff preview tables, and syntax-highlighted source code chunks.
  - `src/ui/app.py`: Full-featured multi-tab Streamlit application:
    - Tab 1 (💬 Codebase Chat): Conversational interface with codebase scoping (all repos or specific indexed repo), retrieval parameter controls (top-k, context token budget, multi-signal re-ranking, LLM temperature), suggested question pills, grounded markdown answers, collapsible citation panels, syntax-highlighted retrieved code snippets with similarity scores, and execution metrics.
    - Tab 2 (📦 Repositories & Ingestion): Ingestion form supporting branch auto-detection, force re-indexing, and background task execution; indexed codebase explorer with file and chunk metrics, remote Git Merkle tree diff inspection previewing pending changes before re-indexing, one-click incremental sync, and cascading repository deletion.
    - Tab 3 (🔍 Vector Collections): Vector store collection browser displaying chunk counts, sample code chunk inspection with language formatting and line ranges, and collection purge capability.
    - Tab 4 (ℹ️ Architecture & About): RAG pipeline overview, workflow walkthrough, and direct links to FastAPI Swagger UI docs.
  - Responsive Sidebar: Live backend connection indicator, vector store & LLM provider diagnostics, repo counts, and connection settings.
  - CLI runner in `src/ui/main.py`: accepts `--host`, `--port`, `--api-url`, and `--headless` flags to launch Streamlit programmatically.
  - Created root `README.md` documenting architecture, setup, CLI usage, REST API endpoints, Streamlit UI, and testing.
  - Pinned `streamlit==1.66.0` and its dependencies in `requirements.txt`.
  - Comprehensive unit test suite in `tests/test_ui.py` (20 tests covering API client requests, connection errors, HTTP errors, formatting helpers, and CLI argument parsing; all passing in 0.27s). Total project unit test suite: 152 tests passing.

Design options considered:
1. Frontend Architecture (Direct backend module imports vs. HTTP API Client):
   - Option A: Have Streamlit import `IncrementalIndexer`, `RAGEngine`, and `ChromaVectorStore` directly in-process.
   - Option B: Decouple Streamlit as a pure client interacting with the FastAPI backend via `ApiClient`.
   - Choice made: Option B.
   - Why: Decoupling respects service boundaries, avoids duplicating model and vector store memory across multiple processes, allows the frontend and backend to scale or run on different hosts/containers (critical for Phase 11 containerization), and simplifies testing through mockable HTTP calls.
2. UI Connection Fault Tolerance & Offline State:
   - Option A: Hard-fail / crash Streamlit with an unhandled exception if the FastAPI backend is not yet started.
   - Option B: Graceful degradation with clear connection diagnostics, offline status banner, and actionable instructions (`python -m src.api.main`).
   - Choice made: Option B.
   - Why: In multi-process and microservice setups, the UI may start before the backend or the backend may temporarily restart. Graceful degradation provides immediate feedback and prevents cryptic stack traces for users and developers.
3. Grounded Code Citation & Chunk Inspection Presentation:
   - Option A: Dump raw chunk text into the assistant message body.
   - Option B: Keep the assistant message concise with clean markdown text, and provide collapsible, syntax-highlighted expanders for citations and source chunks with line numbers and similarity scores.
   - Choice made: Option B.
   - Why: Dumping 10 retrieved code chunks directly into the message body overwhelms the chat feed and makes reading answers difficult. Collapsible accordion panels keep the conversation readable while maintaining 100% transparency and auditability for code verification.

## Phase 10: Evaluation Framework — Complete
What was built:
- Dedicated `src/evaluation/` package for measuring retrieval accuracy, citation groundedness, and hallucination / abstention rates:
  - `src/evaluation/models.py`: Data models `BenchmarkItem`, `RetrievalMetrics`, `GenerationMetrics`, `ItemEvalResult`, and `EvaluationReport` featuring formatted summary tables, GitHub Flavored Markdown reports, and JSON export.
  - `src/evaluation/metrics.py`: Normalized path matching, `compute_retrieval_metrics` (Hit Rate@K, MRR, Precision@K, Recall@K), `compute_citation_metrics` (Citation precision, Citation recall, Hallucinated citations count, Citation groundedness), `compute_symbol_coverage` (codebase symbol presence), `evaluate_abstention` (hallucination resistance and abstention detection on negative probe queries), and composite overall quality scoring.
  - `src/evaluation/benchmark.py`: Curated benchmark suite for `miguelgrinberg/microblog` across authentication, database models, password reset email, application factory, configuration, error handling, route pagination, and negative adversarial probes (`microblog-neg-stripe`, `microblog-neg-k8s`). Dataset loader & saver with JSON persistence in `src/evaluation/datasets/microblog_benchmark.json`.
  - `src/evaluation/evaluator.py`: `CodebaseEvaluator` orchestrating automated benchmark evaluations across vector stores (Chroma, InMemory) and LLM providers (Fake, Gemini), with resilient exception handling and category performance breakdowns.
  - `src/evaluation/main.py`: Command-line evaluation runner (`python -m src.evaluation.main`) supporting `--repo`, `--collection`, `--benchmark`, `--top-k`, `--retrieval-only`, `--llm`, `--output`, and `--markdown`.
- Executed full evaluation run against indexed `miguelgrinberg/microblog` repository in Chroma:
  - Total Queries: 12 (10 targeted functional queries + 2 adversarial negative probes)
  - Retrieval Hit Rate @ 10: 100.0%
  - Mean Reciprocal Rank (MRR): 0.423
  - Retrieval Precision @ 10: 12.0%
  - Retrieval Recall @ 10: 85.0%
  - Citation Groundedness: 100.0% (0 hallucinated citations)
  - Negative Probe Abstention Accuracy: 100.0%
  - Hallucination Rate: 0.0%
  - Overall Quality Score: 76.2%
  - Mean Latency: 334.1 ms
- Comprehensive unit test suite `tests/test_evaluation.py` (17 tests covering metric math, edge cases, model serialization, offline end-to-end evaluation, report formatting, CLI parsing; all passing in ~8s offline). Total project unit test suite: 169 tests passing.
- Integration test suite `tests/test_evaluation_integration.py` verifying live Chroma retrieval evaluation.

Design options considered:
1. Ground truth representation for codebase evaluation:
   - Option A: Exact textual string similarity (ROUGE / BLEU) against human reference answers.
   - Option B: Multi-dimensional evaluation combining file retrieval accuracy (Hit Rate, MRR, Precision@K), citation grounding fidelity (citations mapped to retrieved chunks), code symbol coverage, and explicit abstention/hallucination detection.
   - Choice made: Option B.
   - Why: Codebase Q&A differs fundamentally from conversational chat: answers can be phrased in dozens of syntactically valid ways, rendering n-gram overlap metrics (BLEU/ROUGE) brittle and uninformative. Developers care primarily about: (1) Did retrieval find the right files? (2) Are cited file paths and line numbers real and in the context (zero hallucinated citations)? (3) Are key functions/classes named accurately? (4) Does the model decline to answer when features do not exist? Multi-dimensional metrics directly measure these engineering criteria.
2. Negative probe / hallucination testing strategy:
   - Option A: Only test queries where the target code exists in the repository.
   - Option B: Include explicit out-of-scope / negative probe queries (e.g. asking for Stripe payments or Kubernetes manifests in a lightweight Flask blogging app) with automated abstention detection heuristics.
   - Choice made: Option B.
   - Why: Hallucination is one of the highest risks in RAG systems. If a model invents plausible-sounding files and code when asked about non-existent features, developers lose trust. Incorporating negative probes directly measures whether the system correctly reports feature absence.
3. Offline testability vs. live evaluation:
   - Option A: Only allow evaluation against live LLM API endpoints and real HuggingFace models.
   - Option B: Decouple evaluation metrics and evaluation orchestrator so unit tests run deterministically with `FakeLLMProvider` and `InMemoryVectorStore`, while providing the CLI and integration tests for live runs.
   - Choice made: Option B.
   - Why: Offline testability is a core project standard (unit tests must not need network access, API keys, or the embedding model download). Fast deterministic tests prevent CI breakage and speed up local development while keeping integration tests available under `@pytest.mark.integration`.

## Deferred to later phases
- Caching repeated ingestion (Phase 8 API caching)
- Async/parallel file fetching (optimization, post-MVP)
- Post-ingestion transparency/audit log showing what was filtered and why, for user inspection (Phase 11/12)