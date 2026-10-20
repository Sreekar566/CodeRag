"""
Unit tests for RAG pipeline, context formatting, citation extraction, and RAGEngine.
Uses FakeLLMProvider and mock/in-memory vector storage so no network or API keys are required.
"""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.rag.citations import Citation, extract_citations_from_text, citations_from_chunks
from src.rag.prompt import format_code_context, build_rag_prompt, DEFAULT_SYSTEM_PROMPT
from src.rag.engine import RAGEngine, RAGResponse, ask_codebase
from src.llm.fake import FakeLLMProvider
from src.storage.memory_store import InMemoryVectorStore


class TestPromptAndCitations:
    def test_format_code_context(self):
        chunks = [
            {
                "path": "app/auth/routes.py",
                "start_line": 10,
                "end_line": 25,
                "language": "python",
                "text": "def login():\n    return 'login page'",
            },
            {
                "path": "app/models.py",
                "start_line": 40,
                "end_line": 60,
                "language": "python",
                "text": "class User(db.Model):\n    id = db.Column(db.Integer)",
            },
        ]
        context = format_code_context(chunks)
        assert "[Source 1]" in context
        assert "File: app/auth/routes.py (Lines 10-25)" in context
        assert "```python" in context
        assert "def login():" in context
        assert "[Source 2]" in context
        assert "File: app/models.py (Lines 40-60)" in context

    def test_format_code_context_empty(self):
        assert format_code_context([]) == "No relevant code context found."

    def test_build_rag_prompt(self):
        chunks = [
            {
                "path": "config.py",
                "start_line": 1,
                "end_line": 15,
                "language": "python",
                "text": "SECRET_KEY = os.environ.get('SECRET_KEY')",
            }
        ]
        prompt = build_rag_prompt("How is SECRET_KEY configured?", chunks)
        assert "User Question: How is SECRET_KEY configured?" in prompt
        assert "File: config.py (Lines 1-15)" in prompt
        assert "SECRET_KEY = os.environ.get('SECRET_KEY')" in prompt

    def test_extract_citations_from_text(self):
        text = (
            "Authentication logic is defined in `app/auth/routes.py:12-35`.\n"
            "The user model is in [app/models.py:L40-L50].\n"
            "Main entrypoint is `run.py:5`.\n"
            "See https://flask.palletsprojects.com:80 for details."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 3

        c1 = citations[0]
        assert c1.path == "app/auth/routes.py"
        assert c1.start_line == 12
        assert c1.end_line == 35
        assert c1.format_citation() == "app/auth/routes.py:12-35"

        c2 = citations[1]
        assert c2.path == "app/models.py"
        assert c2.start_line == 40
        assert c2.end_line == 50

        c3 = citations[2]
        assert c3.path == "run.py"
        assert c3.start_line == 5
        assert c3.end_line == 5
        assert c3.format_citation() == "run.py:5"

    def test_citations_from_chunks(self):
        chunks = [
            {
                "path": "app/email.py",
                "start_line": 1,
                "end_line": 20,
                "language": "python",
                "text": "def send_email(): pass",
            }
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 1
        assert citations[0].path == "app/email.py"
        assert citations[0].start_line == 1
        assert citations[0].end_line == 20
        assert citations[0].language == "python"


class TestRAGEngine:
    def test_empty_question_handling(self):
        engine = RAGEngine(vector_store=MagicMock(), llm_provider=FakeLLMProvider())
        resp = engine.answer_question("")
        assert "No question was provided" in resp.answer
        assert resp.citations == []
        assert resp.sources == []
        assert resp.metadata.get("status") == "empty_query"

    def test_empty_collection_handling(self):
        mock_store = MagicMock()
        mock_store.query.return_value = []
        fake_llm = FakeLLMProvider()

        engine = RAGEngine(vector_store=mock_store, llm_provider=fake_llm)
        resp = engine.answer_question("Where is the database initialized?", collection_name="empty_repo")

        assert "No relevant code context was found in collection 'empty_repo'" in resp.answer
        assert resp.citations == []
        assert resp.sources == []
        assert fake_llm.call_count == 0  # Does not waste LLM call when no chunks retrieved

    def test_end_to_end_rag_with_fake_llm(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "id": "app/auth/routes.py:10-30",
                "text": "@bp.route('/login', methods=['GET', 'POST'])\ndef login():\n    pass",
                "path": "app/auth/routes.py",
                "start_line": 10,
                "end_line": 30,
                "language": "python",
                "similarity": 0.88,
            }
        ]
        fake_llm = FakeLLMProvider()

        engine = RAGEngine(vector_store=mock_store, llm_provider=fake_llm)
        resp = engine.answer_question(
            question="Where is login route handled?",
            collection_name="miguelgrinberg_microblog",
        )

        assert isinstance(resp, RAGResponse)
        assert "app/auth/routes.py:10-30" in resp.answer
        assert len(resp.citations) == 1
        assert resp.citations[0].path == "app/auth/routes.py"
        assert resp.citations[0].start_line == 10
        assert resp.citations[0].end_line == 30
        assert len(resp.sources) == 1
        assert resp.model == "fake-model"
        assert fake_llm.call_count == 1
        assert "User Question: Where is login route handled?" in fake_llm.history[0]["prompt"]

    def test_canned_llm_citations_in_response(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "path": "app/models.py",
                "start_line": 1,
                "end_line": 40,
                "text": "class User: pass",
                "language": "python",
            }
        ]
        canned_answer = (
            "The User model is defined in app/models.py:1-40 with password hashing methods."
        )
        fake_llm = FakeLLMProvider(canned_response=canned_answer)

        engine = RAGEngine(vector_store=mock_store, llm_provider=fake_llm)
        resp = engine.answer_question(
            question="Where is User model?",
            repo="miguelgrinberg/microblog",
        )

        assert resp.answer == canned_answer
        assert len(resp.citations) == 1
        assert resp.citations[0].path == "app/models.py"
        assert resp.citations[0].start_line == 1
        assert resp.citations[0].end_line == 40
        assert resp.metadata["collection"] == "miguelgrinberg_microblog"

    def test_fallback_to_chunk_citations_when_llm_omits_lines(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "path": "app/auth/routes.py",
                "start_line": 5,
                "end_line": 25,
                "language": "python",
                "text": "bp = Blueprint('auth', __name__)",
            }
        ]
        # LLM answers without line citation format
        fake_llm = FakeLLMProvider(canned_response="The auth blueprint is initialized in the auth module.")

        engine = RAGEngine(vector_store=mock_store, llm_provider=fake_llm)
        resp = engine.answer_question("Where is the auth blueprint initialized?")

        # Falls back to retrieved source chunk citation
        assert len(resp.citations) == 1
        assert resp.citations[0].path == "app/auth/routes.py"
        assert resp.citations[0].start_line == 5
        assert resp.citations[0].end_line == 25

    def test_format_cited_answer(self):
        resp = RAGResponse(
            query="test query",
            answer="Here is the answer.",
            citations=[
                Citation(path="app/routes.py", start_line=10, end_line=20),
                Citation(path="config.py", start_line=5, end_line=15),
            ],
            sources=[],
            model="fake-model",
        )
        formatted = resp.format_cited_answer()
        assert "Here is the answer." in formatted
        assert "**Citations:**" in formatted
        assert "- `app/routes.py:10-20`" in formatted
        assert "- `config.py:5-15`" in formatted

    def test_ask_codebase_helper(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "path": "run.py",
                "start_line": 1,
                "end_line": 10,
                "language": "python",
                "text": "app.run()",
            }
        ]
        fake_llm = FakeLLMProvider(canned_response="App execution starts in run.py:1-10.")

        resp = ask_codebase(
            question="Where does execution start?",
            repo="test_repo",
            vector_store=mock_store,
            llm_provider=fake_llm,
        )

        assert isinstance(resp, RAGResponse)
        assert resp.citations[0].path == "run.py"
        assert resp.citations[0].start_line == 1
        assert resp.citations[0].end_line == 10

    def test_rag_engine_context_budget_and_metadata(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "path": f"app/file_{i}.py",
                "start_line": 1,
                "end_line": 20,
                "language": "python",
                "text": f"def test_{i}():\n    return {i}",
                "similarity": 0.9 - (i * 0.05),
            }
            for i in range(10)
        ]
        fake_llm = FakeLLMProvider(canned_response="Answer based on file_0.")

        engine = RAGEngine(
            vector_store=mock_store,
            llm_provider=fake_llm,
            max_context_tokens=120,
        )
        resp = engine.answer_question("What files exist?")

        assert resp.metadata["retrieved_count"] == 10
        assert resp.metadata["selected_count"] < 10
        assert resp.metadata["dropped_chunks_count"] > 0
        assert resp.metadata["context_tokens"] <= 150
        assert resp.metadata["reranked"] is True

    def test_rag_engine_merges_overlapping_chunks(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "path": "app/auth/routes.py",
                "start_line": 10,
                "end_line": 20,
                "language": "python",
                "text": "line10\nline11\nline12\nline13\nline14\nline15\nline16\nline17\nline18\nline19\nline20",
                "similarity": 0.8,
            },
            {
                "path": "app/auth/routes.py",
                "start_line": 18,
                "end_line": 25,
                "language": "python",
                "text": "line18\nline19\nline20\nline21\nline22\nline23\nline24\nline25",
                "similarity": 0.85,
            },
        ]
        fake_llm = FakeLLMProvider()

        engine = RAGEngine(vector_store=mock_store, llm_provider=fake_llm)
        resp = engine.answer_question("How is auth routed?")

        # The two overlapping chunks should be merged into 1 chunk covering lines 10 to 25
        assert len(resp.sources) == 1
        assert resp.sources[0]["start_line"] == 10
        assert resp.sources[0]["end_line"] == 25
        assert "line10" in resp.sources[0]["text"]
        assert "line25" in resp.sources[0]["text"]
