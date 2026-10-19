# Progress Handoff

## Completed This Run
- Installed and pinned `pytest==9.1.1`, `iniconfig==2.3.1`, and `pluggy==1.7.0` in `requirements.txt`.
- Updated `.gitignore` to include `data/`, `.pytest_cache/`, and `.cache/`.
- Configured `pytest.ini` with `@pytest.mark.integration` marker (skipped by default).
- Implemented language detection (`detect_language`) in `src/ingestion/chunker.py` and ensured language metadata on all chunks.
- Refactored `src/embedding/embedder.py` with lazy model initialization (`get_model()`) and dependency injection for swappable/mock models.
- Wrote unit tests for `filters.py`, `chunker.py`, `github_client.py` (mocked), and `embedder.py` (mocked).
- Added integration test verifying real 768-dim embeddings from `flax-sentence-embeddings/st-codesearch-distilroberta-base`.
- Verified all 23 unit tests pass in ~4.4s and integration test passes.
- Completed Phase 3; updated `ROADMAP.md` and `project_log.md`.

## Next Steps (Prioritized)
1. **Phase 4 — Vector DB Storage**: Build Chroma wrapper in `src/storage/` with swappable interface storing in `./data/chroma/`.
2. **Phase 4 — Semantic Retrieval**: Implement collection upsert for chunk embeddings + metadata, and similarity search (`query(text, top_k)`).
3. **Phase 4 — Tests**: Add unit tests for vector store operations (ephemeral/mocked) and integration test with real Chroma client.
4. **Phase 5 — LLM Interface**: Implement `src/llm/` with swappable provider interface (`fake` provider for tests, `gemini` via `google-genai`).
5. **Phase 5 — RAG Pipeline**: Connect retrieval to LLM prompt generation to answer natural-language questions with cited file paths and line ranges.
