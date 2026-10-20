"""
Integration tests for Gemini LLM provider and end-to-end RAG pipeline with real models.
Skipped by default unless explicitly invoked with: pytest -m integration
Requires GEMINI_API_KEY environment variable.
"""
from __future__ import annotations

import os
import pytest

from google.genai import errors

from src.llm.gemini import GeminiLLMProvider
from src.rag.engine import RAGEngine
from src.storage.memory_store import InMemoryVectorStore


@pytest.mark.integration
def test_gemini_llm_live_generation():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")

    provider = GeminiLLMProvider(api_key=api_key)
    try:
        response = provider.generate("In 10 words or less, what is RAG?")
    except (errors.ClientError, errors.ServerError) as e:
        pytest.skip(f"Live Gemini API unavailable or quota reached: {e}")

    assert response.content
    assert len(response.content.strip()) > 0
    assert response.model == provider.model


@pytest.mark.integration
def test_rag_engine_live_end_to_end():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")

    # In-memory vector store with sample code chunk
    vector_store = InMemoryVectorStore()
    chunk = {
        "text": "@bp.route('/login', methods=['GET', 'POST'])\ndef login():\n    if current_user.is_authenticated:\n        return redirect(url_for('main.index'))",
        "path": "app/auth/routes.py",
        "start_line": 15,
        "end_line": 25,
        "language": "python",
        "embedding": [0.1] * 768,
    }
    vector_store.add_chunks([chunk], collection_name="test_live_col")

    # Query with dummy embedding function returning match
    vector_store._embed_fn = lambda text: [0.1] * 768

    provider = GeminiLLMProvider(api_key=api_key)
    engine = RAGEngine(vector_store=vector_store, llm_provider=provider)

    try:
        resp = engine.answer_question(
            question="Where is login route handled?",
            collection_name="test_live_col",
        )
    except (errors.ClientError, errors.ServerError) as e:
        pytest.skip(f"Live Gemini API unavailable or quota reached: {e}")

    assert resp.answer
    assert resp.sources
    assert resp.model == provider.model
    assert len(resp.citations) >= 1
