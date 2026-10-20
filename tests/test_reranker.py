"""
Unit tests for query keyword extraction, multi-signal re-ranking, and chunk deduplication.
"""
import pytest

from src.rag.reranker import (
    extract_query_keywords,
    compute_lexical_score,
    compute_path_score,
    compute_definition_score,
    merge_overlapping_chunks,
    ReRanker,
    rerank_chunks,
)


class TestReRanker:
    def test_extract_query_keywords(self):
        query = "How does User.check_password verify hashed passwords in app/models.py?"
        keywords = extract_query_keywords(query)
        # Should contain parsed parts and lowercased tokens
        assert "user" in keywords
        assert "password" in keywords
        assert "passwords" in keywords
        assert "verify" in keywords
        assert "hashed" in keywords
        assert "models" in keywords
        # Stopwords like "how", "does", "in" should be filtered
        assert "how" not in keywords
        assert "does" not in keywords
        assert "in" not in keywords

    def test_extract_camel_and_snake_case(self):
        query = "handleUserLogin and parse_auth_header"
        keywords = extract_query_keywords(query)
        assert "user" in keywords
        assert "login" in keywords
        assert "parse" in keywords
        assert "auth" in keywords
        assert "header" in keywords

    def test_compute_lexical_score(self):
        text = "def check_password(self, password):\n    return check_password_hash(self.password_hash, password)"
        keywords = ["check", "password", "hash"]
        score = compute_lexical_score(text, keywords)
        assert score == 1.0

        keywords_partial = ["check", "password", "unrelated"]
        score_partial = compute_lexical_score(text, keywords_partial)
        assert 0.6 < score_partial < 0.7

    def test_compute_path_score(self):
        path = "app/auth/routes.py"
        keywords = ["auth", "routes", "database"]
        score = compute_path_score(path, keywords)
        assert abs(score - 2 / 3) < 0.01

    def test_compute_definition_score(self):
        python_code = "def authenticate_user(username, password):\n    pass"
        assert compute_definition_score(python_code, ["authenticate_user"]) == 1.0
        assert compute_definition_score(python_code, ["other_func"]) == 0.0

        js_code = "const renderLoginForm = () => { return null; };"
        assert compute_definition_score(js_code, ["renderloginform"]) == 1.0

        class_code = "class DatabaseConnection:\n    pass"
        assert compute_definition_score(class_code, ["databaseconnection"]) == 1.0

    def test_merge_overlapping_chunks(self):
        chunk1 = {
            "path": "app/auth/routes.py",
            "start_line": 10,
            "end_line": 15,
            "text": "line10\nline11\nline12\nline13\nline14\nline15",
            "similarity": 0.8,
        }
        chunk2 = {
            "path": "app/auth/routes.py",
            "start_line": 14,
            "end_line": 18,
            "text": "line14\nline15\nline16\nline17\nline18",
            "similarity": 0.85,
        }
        merged = merge_overlapping_chunks([chunk1, chunk2])
        assert len(merged) == 1
        m = merged[0]
        assert m["path"] == "app/auth/routes.py"
        assert m["start_line"] == 10
        assert m["end_line"] == 18
        assert m["similarity"] == 0.85
        assert m["text"] == "line10\nline11\nline12\nline13\nline14\nline15\nline16\nline17\nline18"

    def test_merge_adjacent_chunks(self):
        chunk1 = {
            "path": "app/auth/routes.py",
            "start_line": 1,
            "end_line": 5,
            "text": "l1\nl2\nl3\nl4\nl5",
        }
        chunk2 = {
            "path": "app/auth/routes.py",
            "start_line": 6,
            "end_line": 8,
            "text": "l6\nl7\nl8",
        }
        merged = merge_overlapping_chunks([chunk1, chunk2])
        assert len(merged) == 1
        assert merged[0]["start_line"] == 1
        assert merged[0]["end_line"] == 8
        assert merged[0]["text"] == "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8"

    def test_merge_chunks_non_overlapping_separate_files(self):
        c1 = {"path": "a.py", "start_line": 1, "end_line": 10, "text": "a"}
        c2 = {"path": "b.py", "start_line": 1, "end_line": 10, "text": "b"}
        merged = merge_overlapping_chunks([c1, c2])
        assert len(merged) == 2

    def test_reranker_prioritizes_exact_matches_and_definitions(self):
        generic_chunk = {
            "path": "app/utils.py",
            "start_line": 1,
            "end_line": 20,
            "text": "# General utilities\nimport os\nprint('starting')",
            "similarity": 0.85,  # High dense similarity
        }
        target_chunk = {
            "path": "app/auth/routes.py",
            "start_line": 10,
            "end_line": 30,
            "text": "def login():\n    # handles user login and authentication\n    return 'login'",
            "similarity": 0.80,  # Slightly lower semantic score
        }

        reranker = ReRanker()
        results = reranker.rerank("How does login authentication route work?", [generic_chunk, target_chunk])
        assert len(results) == 2
        # Target chunk should be boosted to #1 because of path matching, lexical match, and 'def login' definition match
        assert results[0]["path"] == "app/auth/routes.py"
        assert results[0]["rerank_score"] > results[1]["rerank_score"]
