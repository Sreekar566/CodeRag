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
Open question: which files (.flaskenv, LICENSE, Procfile, Vagrantfile, .gitattributes) should be filtered before chunking, and why
Not yet started: filter_files(), chunk_text(), token-based chunking with overlap

## Deferred to later phases
- Caching repeated ingestion (Phase 6/8)
- Concurrency handling for many users (Phase 7)
- Auto-detect default branch instead of hardcoding "main"
- Async/parallel file fetching (optimization, post-MVP)