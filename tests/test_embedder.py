from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from src.embedding.embedder import embed_chunks, embed_query, embed_texts


def test_embed_chunks_empty():
    assert embed_chunks([]) == []


def test_embed_chunks_mocked():
    mock_model = MagicMock()
    # Mock model encoding 2 texts returning numpy arrays of dimension 4
    mock_model.encode.return_value = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ])

    chunks = [
        {"path": "app/routes.py", "start_line": 1, "end_line": 20, "text": "def route(): pass"},
        {"path": "src/App.tsx", "start_line": 1, "end_line": 15, "text": "export const App = () => null;"},
    ]

    result = embed_chunks(chunks, model=mock_model)

    assert len(result) == 2
    mock_model.encode.assert_called_once_with([
        "def route(): pass",
        "export const App = () => null;",
    ])

    # Check that language was auto-detected from path
    assert result[0]["language"] == "python"
    assert result[1]["language"] == "typescript"

    # Check embeddings are lists, not numpy arrays (JSON serializable)
    assert isinstance(result[0]["embedding"], list)
    assert isinstance(result[1]["embedding"], list)
    assert result[0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert result[1]["embedding"] == [0.5, 0.6, 0.7, 0.8]


def test_embed_chunks_preserves_existing_language():
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2]]

    chunks = [
        {
            "path": "app/unknown.xyz",
            "start_line": 1,
            "end_line": 5,
            "text": "sample text",
            "language": "custom_lang",
        }
    ]

    result = embed_chunks(chunks, model=mock_model)
    assert result[0]["language"] == "custom_lang"
    assert result[0]["embedding"] == [0.1, 0.2]


def test_embed_query_mocked():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    vec = embed_query("how is login handled?", model=mock_model)
    assert vec == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once_with("how is login handled?")


def test_embed_query_empty_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        embed_query("")


def test_embed_texts_mocked():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    vecs = embed_texts(["text 1", "text 2"], model=mock_model)
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert embed_texts([], model=mock_model) == []


@pytest.mark.integration
def test_embed_chunks_integration():
    chunks = [
        {
            "path": "app/models.py",
            "start_line": 1,
            "end_line": 5,
            "text": "class User:\n    pass\n",
        }
    ]
    result = embed_chunks(chunks)
    assert len(result) == 1
    assert result[0]["language"] == "python"
    assert isinstance(result[0]["embedding"], list)
    assert len(result[0]["embedding"]) == 768
