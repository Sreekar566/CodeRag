"""
Unit tests for Citation data structures and citation extraction utilities.
Tests Citation formatting, dictionary serialization, text regex extraction,
URL and domain filtering, path normalization, and chunk citation generation.
"""
from __future__ import annotations

import pytest

from src.rag.citations import (
    Citation,
    extract_citations_from_text,
    citations_from_chunks,
    CITATION_REGEX,
    NON_CODE_DOMAIN_SUFFIXES,
)


class TestCitation:
    """Tests for the Citation dataclass and formatting methods."""

    def test_citation_defaults(self):
        c = Citation(path="app/auth.py")
        assert c.path == "app/auth.py"
        assert c.start_line is None
        assert c.end_line is None
        assert c.language is None
        assert c.snippet is None
        assert c.repo is None

    def test_format_citation_line_range(self):
        c = Citation(path="app/auth/routes.py", start_line=10, end_line=25)
        assert c.format_citation() == "app/auth/routes.py:10-25"

    def test_format_citation_single_line_equal(self):
        c = Citation(path="app/models.py", start_line=42, end_line=42)
        assert c.format_citation() == "app/models.py:42"

    def test_format_citation_start_line_only(self):
        c = Citation(path="app/models.py", start_line=42, end_line=None)
        assert c.format_citation() == "app/models.py:42"

    def test_format_citation_no_line_numbers(self):
        c = Citation(path="README.md")
        assert c.format_citation() == "README.md"

    def test_format_citation_with_repo(self):
        c = Citation(path="app/auth.py", start_line=10, end_line=20, repo="miguelgrinberg/microblog")
        assert c.format_citation() == "miguelgrinberg/microblog:app/auth.py:10-20"

    def test_format_citation_with_repo_redundant_path(self):
        c = Citation(
            path="miguelgrinberg/microblog/app/auth.py",
            start_line=10,
            end_line=20,
            repo="miguelgrinberg/microblog",
        )
        assert c.format_citation() == "miguelgrinberg/microblog/app/auth.py:10-20"

    def test_format_citation_with_repo_single_line(self):
        c = Citation(path="app/auth.py", start_line=15, repo="org/repo")
        assert c.format_citation() == "org/repo:app/auth.py:15"

    def test_format_citation_with_repo_no_lines(self):
        c = Citation(path="config.py", repo="org/repo")
        assert c.format_citation() == "org/repo:config.py"

    def test_citation_to_dict(self):
        c = Citation(
            path="src/engine.py",
            start_line=50,
            end_line=75,
            language="python",
            snippet="def execute(): pass",
            repo="owner/project",
        )
        d = c.to_dict()
        assert d == {
            "path": "src/engine.py",
            "start_line": 50,
            "end_line": 75,
            "language": "python",
            "formatted": "owner/project:src/engine.py:50-75",
            "snippet": "def execute(): pass",
            "repo": "owner/project",
        }

    def test_citation_str_dunder(self):
        c = Citation(path="run.py", start_line=1, end_line=10)
        assert str(c) == c.format_citation()
        assert str(c) == "run.py:1-10"


class TestExtractCitationsFromText:
    """Tests for extracting citations from LLM-generated text."""

    def test_extract_empty_and_falsy(self):
        assert extract_citations_from_text("") == []
        assert extract_citations_from_text(None) == []

    def test_extract_plain_text_no_citations(self):
        text = "This repository contains a web application built using Flask and SQLAlchemy."
        assert extract_citations_from_text(text) == []

    def test_extract_standard_line_range(self):
        text = "Authentication routes are implemented in app/auth/routes.py:10-25."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].path == "app/auth/routes.py"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 25

    def test_extract_single_line(self):
        text = "The application entry point is defined at run.py:12."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].path == "run.py"
        assert citations[0].start_line == 12
        assert citations[0].end_line == 12
        assert citations[0].format_citation() == "run.py:12"

    def test_extract_l_prefix_range(self):
        text = "Model definitions are located in app/models.py:L15-L30."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].path == "app/models.py"
        assert citations[0].start_line == 15
        assert citations[0].end_line == 30

    def test_extract_l_prefix_single(self):
        text = "Check error handler at app/errors/handlers.py:L40."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].path == "app/errors/handlers.py"
        assert citations[0].start_line == 40
        assert citations[0].end_line == 40

    def test_extract_markdown_backticks_and_brackets(self):
        text = "See `src/storage/base.py:1-40` and [src/storage/chroma_store.py:20-50]."
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].path == "src/storage/base.py"
        assert citations[0].start_line == 1
        assert citations[0].end_line == 40
        assert citations[1].path == "src/storage/chroma_store.py"
        assert citations[1].start_line == 20
        assert citations[1].end_line == 50

    def test_extract_parenthesized(self):
        text = "Check the authentication logic (app/auth/routes.py:10-25)."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].path == "app/auth/routes.py"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 25

    def test_extract_leading_path_normalization(self):
        text = "Found in `./src/app.py:5-15` and `/app/models.py:20-30`."
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].path == "src/app.py"
        assert citations[0].start_line == 5
        assert citations[0].end_line == 15
        assert citations[1].path == "app/models.py"
        assert citations[1].start_line == 20
        assert citations[1].end_line == 30

    def test_extract_deduplication(self):
        text = (
            "Refer to app/auth/routes.py:10-25. "
            "As stated earlier in app/auth/routes.py:10-25, the view logs in users. "
            "Finally, app/auth/routes.py:10-25 returns the response."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].path == "app/auth/routes.py"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 25

    def test_extract_multiple_distinct_spans_same_file(self):
        text = "Login view is at app/auth.py:1-20 and logout view is at app/auth.py:25-40."
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].path == "app/auth.py"
        assert citations[0].start_line == 1
        assert citations[0].end_line == 20
        assert citations[1].path == "app/auth.py"
        assert citations[1].start_line == 25
        assert citations[1].end_line == 40

    def test_extract_ignores_url_schemes(self):
        text = (
            "Visit https://github.com/org/repo/blob/main/app.py:15-30 for code. "
            "Or http://example.com/api/v1.py:10 for documentation. "
            "Download from ftp://archive.example.org/source.py:20. "
            "Assets loaded from //cdn.example.org/script.js:40."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 0

    def test_extract_ignores_non_code_web_domains(self):
        text = (
            "Connect to api.service.com:8080 or data.platform.org:3000. "
            "Also check net.provider.net:9200, cloud.service.io:4000, "
            "univ.school.edu:80, agency.gov:443, biz.startup.co:5000, "
            "firm.tech.biz:80, and news.blog.info:8080."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 0

    def test_extract_files_with_hyphens_underscores_and_dashes(self):
        text = (
            "See `src/my-module/sub-file.py:10-20` and "
            "deep/pkg_name/test_auth.py:5-15."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].path == "src/my-module/sub-file.py"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 20
        assert citations[1].path == "deep/pkg_name/test_auth.py"
        assert citations[1].start_line == 5
        assert citations[1].end_line == 15

    def test_extract_non_extension_files(self):
        text = "Container setup in docker/Dockerfile:1-10 and build tasks in scripts/build:20-35."
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].path == "docker/Dockerfile"
        assert citations[0].start_line == 1
        assert citations[0].end_line == 10
        assert citations[1].path == "scripts/build"
        assert citations[1].start_line == 20
        assert citations[1].end_line == 35

    def test_extract_mixed_markdown_answer(self):
        text = (
            "### Authentication Overview\n\n"
            "The authentication mechanism consists of:\n"
            "1. **Form validation:** Handled in `app/auth/forms.py:15-30` using WTForms.\n"
            "2. **Route handling:** The login endpoint in [app/auth/routes.py:10-25] checks credentials.\n"
            "3. **Password hashing:** The `User` model in `app/models.py:L40-L60` provides `check_password`.\n\n"
            "For full docs see https://flask-login.readthedocs.io:443.\n"
            "A helper script is available at scripts/init_db:5-15."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 4
        paths = [c.path for c in citations]
        assert paths == [
            "app/auth/forms.py",
            "app/auth/routes.py",
            "app/models.py",
            "scripts/init_db",
        ]


class TestCitationsFromChunks:
    """Tests for generating citations from retrieved source chunks."""

    def test_chunks_empty(self):
        assert citations_from_chunks([]) == []

    def test_chunks_top_level_fields(self):
        chunks = [
            {
                "path": "app/auth.py",
                "start_line": 10,
                "end_line": 25,
                "language": "python",
                "text": "def login(): pass",
                "repo": "owner/repo",
            }
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 1
        c = citations[0]
        assert c.path == "app/auth.py"
        assert c.start_line == 10
        assert c.end_line == 25
        assert c.language == "python"
        assert c.snippet == "def login(): pass"
        assert c.repo == "owner/repo"

    def test_chunks_nested_metadata_dict(self):
        chunks = [
            {
                "id": "c1",
                "text": "class User(db.Model): pass",
                "metadata": {
                    "path": "app/models.py",
                    "start_line": 40,
                    "end_line": 60,
                    "language": "python",
                    "repo": "org/backend",
                },
            }
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 1
        c = citations[0]
        assert c.path == "app/models.py"
        assert c.start_line == 40
        assert c.end_line == 60
        assert c.language == "python"
        assert c.snippet == "class User(db.Model): pass"
        assert c.repo == "org/backend"

    def test_chunks_string_line_numbers_coerced(self):
        chunks = [
            {
                "metadata": {
                    "path": "app/email.py",
                    "start_line": "5",
                    "end_line": "15",
                },
                "text": "def send_mail(): pass",
            }
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 1
        assert citations[0].start_line == 5
        assert citations[0].end_line == 15
        assert isinstance(citations[0].start_line, int)
        assert isinstance(citations[0].end_line, int)

    def test_chunks_missing_path_skipped(self):
        chunks = [
            {"text": "no path chunk"},
            {"path": "", "text": "empty path chunk"},
            {"path": "valid.py", "start_line": 1, "end_line": 5, "text": "valid"},
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 1
        assert citations[0].path == "valid.py"

    def test_chunks_missing_line_numbers(self):
        chunks = [
            {"path": "README.md", "text": "# Documentation"}
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 1
        assert citations[0].path == "README.md"
        assert citations[0].start_line is None
        assert citations[0].end_line is None
        assert citations[0].format_citation() == "README.md"

    def test_chunks_none_metadata_handled(self):
        chunks = [
            {"path": "app.py", "metadata": None, "start_line": 1, "end_line": 10, "text": "code"}
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 1
        assert citations[0].path == "app.py"

    def test_chunks_deduplication(self):
        chunk1 = {"path": "app.py", "start_line": 1, "end_line": 10, "text": "def run(): pass"}
        chunk2 = {"path": "app.py", "start_line": 1, "end_line": 10, "text": "def run(): pass"}
        citations = citations_from_chunks([chunk1, chunk2])
        assert len(citations) == 1
        assert citations[0].path == "app.py"

    def test_chunks_multi_repo(self):
        chunk_repo1 = {"path": "main.py", "start_line": 1, "end_line": 5, "repo": "org/frontend"}
        chunk_repo2 = {"path": "main.py", "start_line": 1, "end_line": 5, "repo": "org/backend"}
        citations = citations_from_chunks([chunk_repo1, chunk_repo2])
        assert len(citations) == 2
        assert citations[0].repo == "org/frontend"
        assert citations[1].repo == "org/backend"
        assert citations[0].format_citation() == "org/frontend:main.py:1-5"
        assert citations[1].format_citation() == "org/backend:main.py:1-5"

    def test_extract_with_repo_prefix_slash(self):
        text = "Frontend renders in `myorg/frontend:src/App.tsx:1-20`."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].repo == "myorg/frontend"
        assert citations[0].path == "src/App.tsx"
        assert citations[0].start_line == 1
        assert citations[0].end_line == 20
        assert citations[0].format_citation() == "myorg/frontend:src/App.tsx:1-20"

    def test_extract_with_repo_prefix_single(self):
        text = "Backend handles auth in [backend:app/auth/routes.py:10-25]."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].repo == "backend"
        assert citations[0].path == "app/auth/routes.py"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 25
        assert citations[0].format_citation() == "backend:app/auth/routes.py:10-25"

    def test_extract_multi_repo_no_collision_same_file(self):
        text = (
            "Frontend component in `myorg/frontend:src/index.tsx:1-15`, while "
            "backend entrypoint is in `myorg/backend:src/index.tsx:1-15`."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].repo == "myorg/frontend"
        assert citations[0].path == "src/index.tsx"
        assert citations[1].repo == "myorg/backend"
        assert citations[1].path == "src/index.tsx"

    def test_extract_inverted_line_numbers(self):
        text = "Reversed line range in app/auth/routes.py:25-10."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].start_line == 10
        assert citations[0].end_line == 25

    def test_extract_windows_drive_letter_not_repo(self):
        text = "Local path file at C:/Users/project/main.py:5-15."
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].repo is None
        assert "main.py" in citations[0].path

    def test_extract_root_extensionless_code_files(self):
        text = (
            "Container setup is in Dockerfile:1-20, build pipeline in Makefile:10-35, "
            "and web process defined in Procfile:1-5."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 3
        paths = [c.path for c in citations]
        assert "Dockerfile" in paths
        assert "Makefile" in paths
        assert "Procfile" in paths

        docker_cit = next(c for c in citations if c.path == "Dockerfile")
        assert docker_cit.start_line == 1
        assert docker_cit.end_line == 20
        assert docker_cit.repo is None

    def test_extract_root_extensionless_with_repo_prefix(self):
        text = "Frontend container in myorg/frontend:Dockerfile:1-15, backend in myorg/backend:Makefile:1-30."
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].repo == "myorg/frontend"
        assert citations[0].path == "Dockerfile"
        assert citations[0].start_line == 1
        assert citations[0].end_line == 15
        assert citations[0].format_citation() == "myorg/frontend:Dockerfile:1-15"

        assert citations[1].repo == "myorg/backend"
        assert citations[1].path == "Makefile"
        assert citations[1].start_line == 1
        assert citations[1].end_line == 30
        assert citations[1].format_citation() == "myorg/backend:Makefile:1-30"

    def test_extract_delimited_custom_extensionless(self):
        text = "Run build via `build:1-10` and entrypoint at [entrypoint:5-20]."
        citations = extract_citations_from_text(text)
        assert len(citations) == 2
        assert citations[0].path == "build"
        assert citations[0].start_line == 1
        assert citations[0].end_line == 10
        assert citations[1].path == "entrypoint"
        assert citations[1].start_line == 5
        assert citations[1].end_line == 20

    def test_extract_rejects_plain_english_standalone_words(self):
        text = (
            "As mentioned in Chapter:1-10 and Step:1-5, follow Table:2-4. "
            "However, check `Dockerfile:1-10` for real code."
        )
        citations = extract_citations_from_text(text)
        assert len(citations) == 1
        assert citations[0].path == "Dockerfile"

    def test_citations_from_chunks_safe_line_parsing(self):
        chunks = [
            {"path": "Dockerfile", "start_line": "L10", "end_line": "L25", "text": "FROM python"},
            {"path": "Makefile", "start_line": "invalid", "end_line": "bad", "text": "all: test"},
            {"path": "run.py", "start_line": 15, "end_line": 5, "text": "print(1)"},
        ]
        citations = citations_from_chunks(chunks)
        assert len(citations) == 3
        # L10-L25 parsed to int 10-25
        assert citations[0].path == "Dockerfile"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 25
        # invalid line strings handled gracefully
        assert citations[1].path == "Makefile"
        assert citations[1].start_line is None
        assert citations[1].end_line is None
        # inverted lines swapped
        assert citations[2].path == "run.py"
        assert citations[2].start_line == 5
        assert citations[2].end_line == 15
