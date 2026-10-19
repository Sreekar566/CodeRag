import pytest
from src.ingestion.chunker import chunk_text, detect_language, CHUNK_SIZE, OVERLAP


def test_detect_language():
    assert detect_language("app/routes.py") == "python"
    assert detect_language("frontend/index.js") == "javascript"
    assert detect_language("frontend/App.jsx") == "javascript"
    assert detect_language("src/types.ts") == "typescript"
    assert detect_language("src/components/Header.tsx") == "typescript"
    assert detect_language("templates/index.html") == "html"
    assert detect_language("static/css/style.css") == "css"
    assert detect_language("README.md") == "markdown"
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("Makefile") == "makefile"
    assert detect_language("Procfile") == "yaml"
    assert detect_language("Vagrantfile") == "ruby"
    assert detect_language("deploy.sh") == "shell"
    assert detect_language("schema.sql") == "sql"
    assert detect_language("config.yaml") == "yaml"
    assert detect_language("data.json") == "json"
    assert detect_language("unknown_file.xyz_unknown") == "unknown"


def test_chunk_text_empty_input():
    assert chunk_text("", "test.py") == []
    assert chunk_text("\n\n\n", "test.py") == []


def test_chunk_text_short_file():
    code = (
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n"
    )
    chunks = chunk_text(code, "math_utils.py")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["path"] == "math_utils.py"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 3
    assert chunk["text"] == code.strip()
    assert chunk["language"] == "python"


def test_chunk_text_sliding_window_multiple_chunks():
    # Generate text with more words than CHUNK_SIZE
    # Each line has 10 words. 60 lines = 600 words > CHUNK_SIZE (400)
    lines = [f"line {i} with several test words in order here" for i in range(1, 61)]
    text = "\n".join(lines)

    chunks = chunk_text(text, "large_file.py")
    assert len(chunks) > 1

    # Check structure of all chunks
    for i, chunk in enumerate(chunks):
        assert chunk["path"] == "large_file.py"
        assert chunk["language"] == "python"
        assert chunk["start_line"] >= 1
        assert chunk["end_line"] >= chunk["start_line"]
        assert len(chunk["text"].strip()) > 0

    # First chunk starts at line 1
    assert chunks[0]["start_line"] == 1
    # Last chunk ends at line 60
    assert chunks[-1]["end_line"] == 60


def test_chunk_text_snaps_to_blank_lines():
    # Build text where a blank line is placed near the boundary
    # CHUNK_SIZE is 400.
    # Lines 1-38 have 10 words each = 380 words.
    # Line 39 is empty.
    # Line 40-42 have 10 words each (total 410 words).
    lines = []
    for i in range(1, 39):
        lines.append(f"word{i}_1 word{i}_2 word{i}_3 word{i}_4 word{i}_5 word{i}_6 word{i}_7 word{i}_8 word{i}_9 word{i}_10")
    lines.append("")  # line 39 is blank
    for i in range(40, 60):
        lines.append(f"word{i}_1 word{i}_2 word{i}_3 word{i}_4 word{i}_5 word{i}_6 word{i}_7 word{i}_8 word{i}_9 word{i}_10")

    text = "\n".join(lines)
    chunks = chunk_text(text, "snap_test.py")

    assert len(chunks) >= 2
    # The first chunk should snap to the blank line at index 38 (line 39)
    assert chunks[0]["end_line"] == 38 or chunks[0]["end_line"] == 39
