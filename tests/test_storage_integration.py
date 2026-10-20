import shutil
import pytest

from src.storage.chroma_store import ChromaVectorStore
from src.embedding.embedder import embed_chunks


@pytest.mark.integration
def test_chroma_persistent_semantic_retrieval(tmp_path):
    chroma_dir = str(tmp_path / "chroma_test")
    store = ChromaVectorStore(persist_directory=chroma_dir, default_collection="integration_test")

    chunks = [
        {
            "path": "app/auth.py",
            "start_line": 1,
            "end_line": 15,
            "text": "def verify_password(stored_password_hash, provided_password):\n    return check_password_hash(stored_password_hash, provided_password)\n",
        },
        {
            "path": "app/routes.py",
            "start_line": 1,
            "end_line": 15,
            "text": "def home_page():\n    return render_template('index.html')\n",
        },
        {
            "path": "app/db.py",
            "start_line": 1,
            "end_line": 15,
            "text": "def init_db():\n    db.create_all()\n",
        },
    ]

    # Embed using the real model
    embedded = embed_chunks(chunks)
    assert len(embedded[0]["embedding"]) == 768

    # Store in persistent Chroma
    stored = store.add_chunks(embedded, collection_name="integration_test")
    assert stored == 3
    assert store.count("integration_test") == 3

    # Query with natural language question
    results = store.query(
        "how to check user password and authentication hash",
        top_k=1,
        collection_name="integration_test",
    )

    assert len(results) == 1
    assert results[0]["path"] == "app/auth.py"
    assert "verify_password" in results[0]["text"]
    assert results[0]["language"] == "python"

    # Cleanup
    shutil.rmtree(chroma_dir, ignore_errors=True)
