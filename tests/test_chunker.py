import pytest
from src.ingestion.chunker import (
    CodeChunker,
    chunk_text,
    detect_language,
    compute_cumulative_words,
    find_line_index_for_word,
    snap_boundary_to_blank_line,
    build_chunk,
    CHUNK_SIZE,
    OVERLAP,
    BOUNDARY_SEARCH_WINDOW,
)


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
    assert chunk_text("   \t  \n  ", "test.py") == []


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


def test_compute_cumulative_words():
    lines = ["one two", "three", "", "four five six"]
    # cumulative should be:
    # 0 words before line 0
    # 2 words before line 1 ("one two")
    # 3 words before line 2 ("three")
    # 3 words before line 3 ("")
    # 6 words in entire file (sentinel)
    expected = [0, 2, 3, 3, 6]
    assert compute_cumulative_words(lines) == expected


def test_find_line_index_for_word():
    cum = [0, 10, 20, 30, 40]
    assert find_line_index_for_word(cum, 0) == 0
    assert find_line_index_for_word(cum, 5) == 1
    assert find_line_index_for_word(cum, 10) == 1
    assert find_line_index_for_word(cum, 11) == 2
    assert find_line_index_for_word(cum, 39) == 4
    assert find_line_index_for_word(cum, 40) == 4


def test_snap_boundary_to_blank_line():
    lines = ["line1", "line2", "", "line4", "line5", "line6"]
    # index 2 is blank ("")
    # if end_line is 3, searching outward within window 2 should find index 2 (offset 1 before)
    assert snap_boundary_to_blank_line(lines, end_line=3, start_line=0, window=2) == 2

    # if window is 0, no snapping
    assert snap_boundary_to_blank_line(lines, end_line=3, start_line=0, window=0) == 3

    # if blank line is before or at start_line, do not collapse
    assert snap_boundary_to_blank_line(lines, end_line=3, start_line=2, window=2) == 3


def test_build_chunk():
    lines = ["def foo():", "    return 42", ""]
    chunk = build_chunk("app/main.py", lines, start_line=0, end_line=2)
    assert chunk["path"] == "app/main.py"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 2
    assert chunk["text"] == "def foo():\n    return 42"
    assert chunk["language"] == "python"

    # Custom language override
    custom_chunk = build_chunk("script.xyz", lines, start_line=0, end_line=2, language="custom_lang")
    assert custom_chunk["language"] == "custom_lang"


def test_code_chunker_custom_parameters():
    # File with 20 lines, 5 words per line = 100 words total
    lines = [f"w{i}_1 w{i}_2 w{i}_3 w{i}_4 w{i}_5" for i in range(1, 21)]
    text = "\n".join(lines)

    chunker = CodeChunker(chunk_size=30, overlap=10, boundary_search_window=2)
    chunks = chunker.chunk(text, "test.py")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["language"] == "python"
        assert chunk["start_line"] >= 1
        assert chunk["end_line"] >= chunk["start_line"]


def test_code_chunker_validation():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        CodeChunker(chunk_size=0)

    with pytest.raises(ValueError, match="overlap must be non-negative"):
        CodeChunker(chunk_size=100, overlap=-5)

    with pytest.raises(ValueError, match="overlap .* must be strictly less than chunk_size"):
        CodeChunker(chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="overlap .* must be strictly less than chunk_size"):
        CodeChunker(chunk_size=100, overlap=150)

    with pytest.raises(ValueError, match="boundary_search_window must be non-negative"):
        CodeChunker(boundary_search_window=-1)


def test_chunk_text_custom_parameters():
    lines = [f"word {i} test" for i in range(1, 40)]
    text = "\n".join(lines)

    chunks = chunk_text(
        text=text,
        path="custom.ts",
        chunk_size=20,
        overlap=5,
        boundary_window=0,
        language="typescript",
    )
    assert len(chunks) > 1
    assert chunks[0]["language"] == "typescript"
