import requests
import base64
import os

GITHUB_API_BASE = "https://api.github.com"


def get_repo_tree(owner: str, repo: str, token: str | None = None, branch: str = "main"):
    """
    Fetch the full recursive file tree for a GitHub repo.

    Returns a list of dicts like:
        {"path": "app/routes.py", "type": "blob", "sha": "..."}
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}"
    params = {"recursive": "1"}
    headers = {"Authorization": f"token {token}"} if token else {}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 404:
        raise ValueError(f"Repo or branch not found: {owner}/{repo}@{branch}")
    if response.status_code == 403:
        raise RuntimeError("Rate limit hit or access forbidden")
    response.raise_for_status()  # catches any other unexpected error

    data = response.json()

    # Only keep files ("blob"), not folders ("tree")
    files = [item for item in data["tree"] if item["type"] == "blob"]
    return files



def get_file_content(owner: str, repo: str, path: str, token: str | None = None):
    """
    Fetch and decode the raw text content of a single file in a repo.

    Returns the file content as a string, or None if it's binary/undecodable.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"} if token else {}

    response = requests.get(url, headers=headers)

    if response.status_code == 404:
        raise ValueError(f"File not found: {path}")
    if response.status_code == 403:
        raise RuntimeError("Rate limit hit or access forbidden")
    response.raise_for_status()

    data = response.json()

    # GitHub returns file content base64-encoded
    raw_bytes = base64.b64decode(data["content"])

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # It's binary (image, compiled file, etc.) - can't treat as text
        return None


def get_default_branch(owner: str, repo: str, token: str | None = None) -> str:
    """
    Fetch the default branch for a GitHub repository.
    Returns the branch name (e.g., 'main' or 'master').
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {token}"} if token else {}

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        raise ValueError(f"Repo not found: {owner}/{repo}")
    if response.status_code == 403:
        raise RuntimeError("Rate limit hit or access forbidden")
    response.raise_for_status()

    data = response.json()
    return data.get("default_branch", "main")


def get_latest_commit_sha(owner: str, repo: str, token: str | None = None, branch: str = "main") -> str:
    """
    Fetch the latest commit SHA for a specific branch of a GitHub repository.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{branch}"
    headers = {"Authorization": f"token {token}"} if token else {}

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        raise ValueError(f"Repo or branch not found: {owner}/{repo}@{branch}")
    if response.status_code == 403:
        raise RuntimeError("Rate limit hit or access forbidden")
    response.raise_for_status()

    data = response.json()
    return data["sha"]


if __name__ == "__main__":
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN environment variable first")

    files = get_repo_tree("miguelgrinberg", "microblog", token)
    print(f"Found {len(files)} files")

    # Test fetching one specific file's content
    content = get_file_content("miguelgrinberg", "microblog", "app/auth/routes.py", token)
    if content is None:
        print("File was binary, could not decode")
    else:
        print(content[:500])  # print first 500 characters only