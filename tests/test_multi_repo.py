"""
Unit tests for multi-repository querying, cross-repo context assembly, and citations.
"""
from __future__ import annotations

import pytest

from src.rag.engine import RAGEngine, ask_codebase
from src.llm.fake import FakeLLMProvider
from src.storage.memory_store import InMemoryVectorStore
from src.registry.registry import RepoRegistry
from src.registry.models import RepositoryMetadata


def test_multi_repo_querying_and_context_assembly():
    dummy_embed = lambda q: [0.5, 0.5]
    store = InMemoryVectorStore(embed_fn=dummy_embed)
    registry = RepoRegistry(storage_path=None)

    # Register repo 1: frontend
    registry.register(
        RepositoryMetadata(
            owner="myorg",
            repo="frontend",
            collection_name="myorg_frontend",
        )
    )
    store.add_chunks(
        [
            {
                "id": "src/App.tsx:1-20",
                "path": "src/App.tsx",
                "start_line": 1,
                "end_line": 20,
                "language": "typescript",
                "text": "function App() {\n  return <LoginButton />;\n}",
                "embedding": [0.8, 0.2],
            }
        ],
        collection_name="myorg_frontend",
    )

    # Register repo 2: backend
    registry.register(
        RepositoryMetadata(
            owner="myorg",
            repo="backend",
            collection_name="myorg_backend",
        )
    )
    store.add_chunks(
        [
            {
                "id": "app/auth/routes.py:1-25",
                "path": "app/auth/routes.py",
                "start_line": 1,
                "end_line": 25,
                "language": "python",
                "text": "def verify_token(token):\n    return decode_jwt(token)",
                "embedding": [0.7, 0.3],
            }
        ],
        collection_name="myorg_backend",
    )

    fake_llm = FakeLLMProvider(
        canned_response=(
            "Frontend renders the button in `src/App.tsx:1-20`, while backend "
            "verifies the token in `app/auth/routes.py:1-25`."
        )
    )

    engine = RAGEngine(
        vector_store=store,
        llm_provider=fake_llm,
        registry=registry,
    )

    response = engine.answer_question(
        "How is user login handled across the stack?",
        repos=["myorg/frontend", "myorg/backend"],
        top_k=5,
    )

    assert response.metadata["multi_repo"] is True
    assert set(response.metadata["repositories"]) == {"myorg/frontend", "myorg/backend"}
    assert response.metadata["retrieved_count"] == 2
    assert response.metadata["selected_count"] == 2

    # Check sources have repo tags
    source_repos = {s.get("repo") for s in response.sources}
    assert source_repos == {"myorg/frontend", "myorg/backend"}

    # Prompt sent to LLM should have repo annotations
    last_prompt = fake_llm.history[-1]["prompt"]
    assert "Repo: myorg/frontend" in last_prompt
    assert "Repo: myorg/backend" in last_prompt

    # Citations extracted
    citation_paths = [c.path for c in response.citations]
    assert "src/App.tsx" in citation_paths
    assert "app/auth/routes.py" in citation_paths


def test_answer_multi_repo_convenience_method():
    dummy_embed = lambda q: [0.1, 0.2]
    store = InMemoryVectorStore(embed_fn=dummy_embed)
    registry = RepoRegistry(storage_path=None)

    registry.register(RepositoryMetadata(owner="org", repo="repo1", collection_name="org_repo1"))
    registry.register(RepositoryMetadata(owner="org", repo="repo2", collection_name="org_repo2"))

    store.add_chunks(
        [{"id": "a.py:1-5", "path": "a.py", "start_line": 1, "end_line": 5, "text": "def func_a(): pass", "embedding": [0.1, 0.2]}],
        collection_name="org_repo1",
    )
    store.add_chunks(
        [{"id": "b.py:1-5", "path": "b.py", "start_line": 1, "end_line": 5, "text": "def func_b(): pass", "embedding": [0.1, 0.2]}],
        collection_name="org_repo2",
    )

    fake_llm = FakeLLMProvider()
    engine = RAGEngine(vector_store=store, llm_provider=fake_llm, registry=registry)

    resp = engine.answer_multi_repo(
        question="What functions exist?",
        repos=["org/repo1", "org/repo2"],
    )

    assert resp.metadata["multi_repo"] is True
    assert len(resp.sources) == 2


def test_multi_repo_empty_results():
    dummy_embed = lambda q: [0.1, 0.2]
    store = InMemoryVectorStore(embed_fn=dummy_embed)
    fake_llm = FakeLLMProvider()

    engine = RAGEngine(vector_store=store, llm_provider=fake_llm)
    resp = engine.answer_question(
        question="Where is the cache?",
        repos=["org/empty1", "org/empty2"],
    )

    assert resp.metadata["retrieved_count"] == 0
    assert "No relevant code context was found in collections 'org_empty1, org_empty2'" in resp.answer
    assert fake_llm.call_count == 0  # Does not hallucinate or waste tokens


def test_ask_codebase_multi_repo():
    dummy_embed = lambda q: [0.5, 0.5]
    store = InMemoryVectorStore(embed_fn=dummy_embed)
    registry = RepoRegistry(storage_path=None)

    store.add_chunks(
        [{"id": "service.py:1-10", "path": "service.py", "start_line": 1, "end_line": 10, "text": "class Service: pass", "embedding": [0.5, 0.5]}],
        collection_name="col_svc",
    )
    fake_llm = FakeLLMProvider()

    resp = ask_codebase(
        question="Show service class",
        collections=["col_svc"],
        vector_store=store,
        llm_provider=fake_llm,
        registry=registry,
    )

    assert resp.metadata["retrieved_count"] == 1
    assert len(resp.sources) == 1
