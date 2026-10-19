# Noise directories - fill these in (folder names that are never authored source)
NOISE_DIRS = {"migrations", "node_modules", "__pycache__",
    ".git", "venv", ".venv", ".pytest_cache", "dist", "build"
}

# Noise extensions - fill these in (file endings that are never authored source)
NOISE_EXTENSIONS = { '.gitignore', '.gitattributes', '.flaskenv', '.gif', '.png', '.jpg', '.jpeg', '.bmp', '.ico', '.svg', '.mp4', '.mp3', '.wav', '.ogg', '.flac', '.avi', '.mov', '.wmv', '.mkv', '.zip', '.tar.gz', '.rar', '.7z', '.po', 'LICENSE'}

# Size threshold in bytes - pick a number and be ready to explain why
MAX_FILE_SIZE = 500_000


def filter_files(tree: list[dict]) -> list[dict]:
    """Takes the list of file dicts from get_repo_tree(), returns only the ones worth chunking/embedding."""
    kept = []
    for f in tree:
        path = f["path"]

        if any(dir_name in path.split("/") for dir_name in NOISE_DIRS):
            continue

        if any(path.endswith(ext) for ext in NOISE_EXTENSIONS):
            continue

        if MAX_FILE_SIZE is not None and f["size"] > MAX_FILE_SIZE:
                    continue
        
        kept.append(f)

    return kept
