# TASK — AI Codebase Intelligence Platform (unattended build)

You are an autonomous agent running on a schedule with no human available.
Never stop to ask questions: make a reasonable decision, write down why in
`project_log.md`, and keep going.

## Goal
Finish the platform described in `ROADMAP.md`: a Python system that ingests
any public GitHub repository and answers natural-language questions about its
code with RAG, giving grounded answers that cite file paths and line ranges.

## Source of truth
- `ROADMAP.md` — phase list, status, tech stack, and working conventions.
  Follow its conventions. Update its Status section as phases are completed.
- `project_log.md` — append an entry for every phase you work on: what was
  built, design options considered, the choice made, and why.
- `PROGRESS.md` — short handoff note between runs: what you did this run and
  the next 3–5 concrete steps. Keep it under 40 lines.

## How to work each run
1. Read `ROADMAP.md`, `PROGRESS.md` and the existing code before changing
   anything. Phases 1–2 are complete; Phase 3 is in progress.
2. Work on the lowest unfinished phase only. Do not start a new phase until
   the current one runs end-to-end and its tests pass.
3. Prefer small, verified steps. Leaving a run with fewer features that work
   is better than more features that don't.
4. Before finishing a run, make sure `pytest` passes.

## Decisions already made (use these, don't re-litigate)
- **Language:** Python 3.12. Keep `requirements.txt` updated with pinned
  versions for every dependency you add.
- **PyTorch:** CPU-only. This server has no GPU and 4 GB RAM. Install with
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **Embedding model:** keep `flax-sentence-embeddings/st-codesearch-distilroberta-base`
  (see `src/embedding/embedder.py`). Do not switch to models that need
  `trust_remote_code`.
- **Vector DB:** Chroma, persistent local storage in `./data/chroma/`
  (gitignored). Wrap it behind a small interface so it could be swapped.
- **LLM:** call it through one interface in `src/llm/` with the provider
  chosen by the `LLM_PROVIDER` env var. Implement `gemini` (via the
  `google-genai` SDK, key in `GEMINI_API_KEY`) and a `fake` provider that
  returns deterministic canned answers. Tests must use `fake` only.
- **Backend:** FastAPI (Phase 8). **Frontend:** Streamlit (Phase 9).
- **Test repo:** `miguelgrinberg/microblog`, as in earlier phases.

## Testing rules
- Tests live in `tests/`, run with `pytest`. Add `pytest` to requirements.
- Unit tests must not need network access, API keys, or the embedding model
  download: mock the GitHub API and the embedder, and use the `fake` LLM.
- Tests that need the network or real models go behind a pytest marker
  (`@pytest.mark.integration`) and are skipped by default.
- First thing to add, before more Phase 3 work: unit tests for
  `filters.py`, `chunker.py`, and `github_client.py` (mocked).

## Hard rules
- Never run git commands; the wrapper script commits and pushes.
- Never write secrets, tokens, or API keys into any file. Read them from
  environment variables only (`GITHUB_TOKEN`, `GEMINI_API_KEY`).
- Never commit data or model files: keep `data/`, model caches, and `.env`
  in `.gitignore`.
- Don't change existing public function signatures without updating every
  caller and test.
- Deployment (Phase 11) means a working `Dockerfile`, `docker-compose.yml`
  and a GitHub Actions CI workflow that runs `pytest`. Don't deploy anywhere.

## Definition of done
Phases 3–12 in `ROADMAP.md` are marked ✅, `pytest` passes, the README
explains how to set up, index a repo, and ask questions (CLI, API, and
Streamlit UI), and the evaluation from Phase 10 has been run against
`microblog` with its results recorded in `project_log.md`. Then make the
first line of `PROGRESS.md` exactly: DONE
