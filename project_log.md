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

## Phase 2: File Filtering and Code Chunking — In Progress
Resolved: filter decision reframed as "does this file help answer a target question type" not "is this code vs config."
- .flaskenv, LICENSE, .gitattributes -> filter out (settings/legal/git-internal, no behavior, not deployment-relevant)
- Procfile, Vagrantfile -> keep (onboarding/deployment questions are in scope for "summarize repo for a new engineer")
Filter design: blocklist strategy chosen over allowlist (generalizes across languages without enumerating every source extension; accepted tradeoff is fail-open risk vs allowlist's fail-closed). Three layers: (1) noise directory blocklist, (2) noise extension/filename blocklist, (3) file size threshold as a red flag for generated/data files. Extensionless files (Dockerfile, Makefile, Procfile) pass through automatically under blocklist - no special-casing needed.
Considered and rejected: human-in-the-loop review of ambiguous/extensionless binary files during ingestion. Rejected because (1) it breaks the automated-pipeline promise and is incompatible with async background workers (Phase 7), and (2) it's redundant - get_file_content() from Phase 1 already hard-fails to decode true binaries, so there's no real ambiguity left for a human to resolve.
In progress: writing filter_files(tree) in src/ingestion/filters.py (tree-metadata layer, no content fetch)
Not yet started: chunk_text(), token-based chunking with overlap

## Deferred to later phases
- Caching repeated ingestion (Phase 6/8)
- Concurrency handling for many users (Phase 7)
- Auto-detect default branch instead of hardcoding "main"
- Async/parallel file fetching (optimization, post-MVP)
- Post-ingestion transparency/audit log showing what was filtered and why, for user inspection (pairs naturally with Phase 11 observability, not a blocking ingestion-time prompt)
