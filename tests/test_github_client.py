import base64
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.ingestion.github_client import (
    get_repo_tree,
    get_file_content,
    get_default_branch,
    get_latest_commit_sha,
)


def test_get_repo_tree_success():
    fake_tree = {
        "tree": [
            {"path": "app/routes.py", "type": "blob", "sha": "abc123"},
            {"path": "app", "type": "tree", "sha": "def456"},
            {"path": "README.md", "type": "blob", "sha": "789xyz"},
        ]
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = fake_tree

    with patch("requests.get", return_value=mock_response) as mock_get:
        files = get_repo_tree("owner", "repo", "fake_token", branch="main")

        assert len(files) == 2
        assert files[0]["path"] == "app/routes.py"
        assert files[1]["path"] == "README.md"

        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/git/trees/main",
            headers={"Authorization": "token fake_token"},
            params={"recursive": "1"},
        )


def test_get_repo_tree_404_raises_value_error():
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="Repo or branch not found"):
            get_repo_tree("owner", "nonexistent", "fake_token")


def test_get_repo_tree_403_raises_runtime_error():
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Rate limit hit or access forbidden"):
            get_repo_tree("owner", "repo", "fake_token")


def test_get_repo_tree_http_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.HTTPError("Server Error")

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(requests.HTTPError):
            get_repo_tree("owner", "repo", "fake_token")


def test_get_file_content_text_success():
    raw_code = "print('hello from codebase intelligence')"
    b64_content = base64.b64encode(raw_code.encode("utf-8")).decode("utf-8")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": b64_content}

    with patch("requests.get", return_value=mock_response) as mock_get:
        content = get_file_content("owner", "repo", "main.py", "fake_token")
        assert content == raw_code
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/main.py",
            headers={"Authorization": "token fake_token"},
        )


def test_get_file_content_binary_returns_none():
    # Non-decodable bytes (invalid UTF-8)
    invalid_utf8 = b"\x80\x81\xff\xfe"
    b64_content = base64.b64encode(invalid_utf8).decode("utf-8")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": b64_content}

    with patch("requests.get", return_value=mock_response):
        content = get_file_content("owner", "repo", "binary.dat", "fake_token")
        assert content is None


def test_get_file_content_404_raises_value_error():
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="File not found"):
            get_file_content("owner", "repo", "missing.py", "fake_token")


def test_get_file_content_403_raises_runtime_error():
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Rate limit hit or access forbidden"):
            get_file_content("owner", "repo", "some_file.py", "fake_token")


def test_get_file_content_http_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.HTTPError("Server Error")

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(requests.HTTPError):
            get_file_content("owner", "repo", "some_file.py", "fake_token")


def test_get_default_branch_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"default_branch": "master", "name": "repo"}

    with patch("requests.get", return_value=mock_response) as mock_get:
        branch = get_default_branch("owner", "repo", "fake_token")
        assert branch == "master"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo",
            headers={"Authorization": "token fake_token"},
        )


def test_get_default_branch_default_fallback():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "repo"}

    with patch("requests.get", return_value=mock_response):
        branch = get_default_branch("owner", "repo")
        assert branch == "main"


def test_get_default_branch_404_raises():
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="Repo not found"):
            get_default_branch("owner", "missing")


def test_get_default_branch_403_raises():
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Rate limit hit or access forbidden"):
            get_default_branch("owner", "repo")


def test_get_latest_commit_sha_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"sha": "c0ffee1234567890", "commit": {"message": "Initial commit"}}

    with patch("requests.get", return_value=mock_response) as mock_get:
        sha = get_latest_commit_sha("owner", "repo", "fake_token", branch="main")
        assert sha == "c0ffee1234567890"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/commits/main",
            headers={"Authorization": "token fake_token"},
        )


def test_get_latest_commit_sha_404_raises():
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="Repo or branch not found"):
            get_latest_commit_sha("owner", "repo", branch="dev")


def test_get_latest_commit_sha_403_raises():
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Rate limit hit or access forbidden"):
            get_latest_commit_sha("owner", "repo")

