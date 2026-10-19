import os

# Chunk size and overlap, in approximate tokens (whitespace-split words as a
# stand-in until an embedding model is chosen and a real tokenizer is wired in).
# Pick numbers and be ready to explain why.
CHUNK_SIZE = 400  # e.g. 500
OVERLAP = 50  # e.g. 50

# How far (in lines) to search for a nearby blank line when snapping a chunk
# boundary, instead of cutting at the exact word count.
BOUNDARY_SEARCH_WINDOW = 5  # e.g. 5

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".less": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".r": "r",
    ".lua": "lua",
    ".xml": "xml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".txt": "text",
}

FILENAME_TO_LANGUAGE = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "vagrantfile": "ruby",
    "procfile": "yaml",
}


def detect_language(path: str) -> str:
    """
    Detect the programming or markup language from a file path based on
    its filename or extension.
    """
    filename = os.path.basename(path).lower()
    if filename in FILENAME_TO_LANGUAGE:
        return FILENAME_TO_LANGUAGE[filename]

    _, ext = os.path.splitext(filename)
    if ext in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext]

    return "unknown"


def chunk_text(text: str, path: str) -> list[dict]:
    """
    Splits a file's text into overlapping chunks, each tagged with the file
    path, language, and the line range it covers.

    Returns a list of dicts like:
        {"path": "app/models.py", "start_line": 1, "end_line": 42, "text": "...", "language": "python"}
    """
    lines = text.split("\n")

    # step 1: figure out, for each line, how many words came before it
    # (a running total) — you'll need this to convert "word N" back into
    # "line number" when you record start_line/end_line for a chunk.
    # hint: build a list where cumulative_words[i] = total words in lines[0:i]
    cumulative_words = []
    for i, line in enumerate(lines):
        if i == 0:
            cumulative_words.append(0)
        else:
            cumulative_words.append(cumulative_words[i-1] + len(lines[i-1].split()))

    # sentinel entry: total words in the whole file, including the last line
    # (cumulative_words would otherwise stop one line short of the true total)
    cumulative_words.append(cumulative_words[-1] + len(lines[-1].split()))
    # step 2: walk forward through the text in CHUNK_SIZE-word windows,
    # stepping by (CHUNK_SIZE - OVERLAP) words each time so consecutive
    # chunks overlap by OVERLAP words.
    total_words = cumulative_words[-1]
    start_word = 0
    chunks = []
    while start_word < total_words:
        end_word = min(start_word + CHUNK_SIZE, total_words)

        # find the line numbers corresponding to start_word and end_word
        start_line = next(i for i, cw in enumerate(cumulative_words) if cw >= start_word)
        end_line = next(i for i, cw in enumerate(cumulative_words) if cw >= end_word)

        # snap the chunk's end to the nearest blank line if BOUNDARY_SEARCH_WINDOW is set
        # search outward from end_line (distance 0, then 1, then 2, ...) so we
        # actually find the closest blank line, not just the first one in the window
        # skip snapping on the final chunk (end_word >= total_words) — there's no
        # next chunk to pick up content that would otherwise get pushed past the cut
        if BOUNDARY_SEARCH_WINDOW is not None and end_word < total_words:
            for offset in range(BOUNDARY_SEARCH_WINDOW + 1):
                before = end_line - offset
                after = end_line + offset
                if 0 <= before < len(lines) and before > start_line and lines[before].strip() == "":
                    end_line = before
                    break
                if after < len(lines) and lines[after].strip() == "":
                    end_line = after
                    break

        # slice the lines for this chunk and join them back into text
        chunk_body = "\n".join(lines[start_line:end_line])

        # build the chunk dict and add it to the list
        chunks.append({
            "path": path,
            "start_line": start_line + 1,  # convert to 1-based line numbers
            "end_line": end_line,
            "text": chunk_body,
            "language": detect_language(path),
        })

        # move to the next chunk window
        start_word += CHUNK_SIZE - OVERLAP

    return chunks