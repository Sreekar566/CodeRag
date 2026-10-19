import pytest
from src.ingestion.filters import filter_files, NOISE_DIRS, NOISE_EXTENSIONS, MAX_FILE_SIZE


def test_filter_keeps_valid_source_files():
    tree = [
        {"path": "app/routes.py", "size": 1200},
        {"path": "app/models.py", "size": 3400},
        {"path": "Dockerfile", "size": 500},
        {"path": "Procfile", "size": 100},
        {"path": "Vagrantfile", "size": 800},
        {"path": "README.md", "size": 2500},
        {"path": "src/components/App.tsx", "size": 4200},
    ]
    kept = filter_files(tree)
    paths = [f["path"] for f in kept]
    assert paths == [
        "app/routes.py",
        "app/models.py",
        "Dockerfile",
        "Procfile",
        "Vagrantfile",
        "README.md",
        "src/components/App.tsx",
    ]


def test_filter_removes_noise_directories():
    tree = [
        {"path": "migrations/versions/001_init.py", "size": 1000},
        {"path": "node_modules/express/index.js", "size": 2000},
        {"path": "app/__pycache__/routes.cpython-312.pyc", "size": 1500},
        {"path": ".git/config", "size": 300},
        {"path": "venv/bin/activate", "size": 400},
        {"path": ".venv/lib/python3.12/site.py", "size": 500},
        {"path": ".pytest_cache/v/cache/lastfailed", "size": 100},
        {"path": "dist/bundle.js", "size": 50000},
        {"path": "build/main.o", "size": 20000},
        {"path": "valid_dir/app.py", "size": 1000},
    ]
    kept = filter_files(tree)
    assert len(kept) == 1
    assert kept[0]["path"] == "valid_dir/app.py"


def test_filter_removes_noise_extensions_and_filenames():
    tree = [
        {"path": ".gitignore", "size": 100},
        {"path": ".gitattributes", "size": 80},
        {"path": ".flaskenv", "size": 50},
        {"path": "static/img/logo.png", "size": 12000},
        {"path": "static/img/banner.jpg", "size": 30000},
        {"path": "static/img/icon.svg", "size": 1500},
        {"path": "assets/sample.mp4", "size": 80000},
        {"path": "backup/code.zip", "size": 45000},
        {"path": "translations/messages.po", "size": 6000},
        {"path": "LICENSE", "size": 1100},
        {"path": "main.py", "size": 1200},
    ]
    kept = filter_files(tree)
    assert len(kept) == 1
    assert kept[0]["path"] == "main.py"


def test_filter_removes_oversized_files():
    tree = [
        {"path": "normal.py", "size": MAX_FILE_SIZE - 1},
        {"path": "exact_limit.py", "size": MAX_FILE_SIZE},
        {"path": "huge_generated_data.json", "size": MAX_FILE_SIZE + 1},
        {"path": "giant_dataset.csv", "size": 10_000_000},
    ]
    kept = filter_files(tree)
    paths = [f["path"] for f in kept]
    assert paths == ["normal.py", "exact_limit.py"]


def test_filter_handles_missing_size_key():
    tree = [
        {"path": "app/models.py"},
        {"path": "migrations/001.py"},
    ]
    kept = filter_files(tree)
    assert len(kept) == 1
    assert kept[0]["path"] == "app/models.py"


def test_filter_handles_empty_tree():
    assert filter_files([]) == []
