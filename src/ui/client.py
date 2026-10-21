"""
API client for interacting with the FastAPI backend of the AI Codebase Intelligence Platform.
Provides strongly-typed methods, clean exception handling, and connection validation.
"""
from __future__ import annotations

import os
from typing import Any
import requests


class ApiClientError(Exception):
    """Base exception for API client errors."""

    def __init__(self, message: str, status_code: int | None = None, details: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class ApiConnectionError(ApiClientError):
    """Exception raised when the client cannot connect to the backend API server."""
    pass


class ApiClient:
    """
    HTTP client for the AI Codebase Intelligence backend API.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        url = base_url or os.getenv("API_BASE_URL") or "http://localhost:8000"
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        req_timeout = timeout if timeout is not None else self.timeout

        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=req_timeout,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            raise ApiConnectionError(
                f"Failed to connect to backend server at '{self.base_url}'. "
                "Ensure the API server is running (e.g., 'python -m src.api.main').",
                details=str(exc),
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"HTTP request error: {exc}", details=str(exc)) from exc

        if resp.status_code >= 400:
            error_message = f"API returned status {resp.status_code}"
            details = None
            try:
                data = resp.json()
                if isinstance(data, dict):
                    error_message = data.get("detail") or data.get("error") or error_message
                    details = data
            except Exception:
                error_message = resp.text or error_message

            raise ApiClientError(error_message, status_code=resp.status_code, details=details)

        if resp.status_code == 204:
            return {}

        try:
            return resp.json()
        except Exception as exc:
            raise ApiClientError(f"Failed to parse JSON response: {exc}") from exc

    def health(self) -> dict[str, Any]:
        """
        Check backend system health and runtime configuration.
        Returns HealthResponse data dict.
        """
        return self._request("GET", "/health", timeout=5.0)

    def is_connected(self) -> bool:
        """
        Quick boolean check whether the backend API is reachable.
        """
        try:
            res = self.health()
            return res.get("status") == "ok"
        except Exception:
            return False

    def list_repos(self) -> list[dict[str, Any]]:
        """
        Retrieve all registered codebases.
        """
        data = self._request("GET", "/repos")
        return data.get("repos", [])

    def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """
        Retrieve metadata for a specific repository.
        """
        return self._request("GET", f"/repos/{owner}/{repo}")

    def delete_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """
        Delete a repository and purge its collections from the vector store.
        """
        return self._request("DELETE", f"/repos/{owner}/{repo}")

    def get_repo_changes(
        self,
        owner: str,
        repo: str,
        branch: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        """
        Inspect pending Git changes between remote GitHub and current index.
        """
        params: dict[str, Any] = {}
        if branch:
            params["branch"] = branch
        if token:
            params["token"] = token
        return self._request("GET", f"/repos/{owner}/{repo}/changes", params=params)

    def index_repo(
        self,
        owner: str,
        repo: str,
        branch: str | None = None,
        token: str | None = None,
        collection_name: str | None = None,
        force: bool = False,
        background: bool = False,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """
        Trigger repository indexing (synchronous or background).
        """
        payload: dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "force": force,
            "background": background,
        }
        if branch is not None:
            payload["branch"] = branch
        if token:
            payload["token"] = token
        if collection_name:
            payload["collection_name"] = collection_name

        return self._request("POST", "/index", json=payload, timeout=timeout)

    def query(
        self,
        question: str,
        repo: str | None = None,
        repos: list[str] | None = None,
        collection_name: str | None = None,
        collections: list[str] | None = None,
        top_k: int = 10,
        max_context_tokens: int | None = None,
        rerank: bool = True,
        merge_overlapping: bool = True,
        temperature: float = 0.2,
        where: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """
        Execute natural language Q&A against the codebase.
        """
        payload: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "rerank": rerank,
            "merge_overlapping": merge_overlapping,
            "temperature": temperature,
        }
        if repo:
            payload["repo"] = repo
        if repos:
            payload["repos"] = repos
        if collection_name:
            payload["collection_name"] = collection_name
        if collections:
            payload["collections"] = collections
        if max_context_tokens is not None:
            payload["max_context_tokens"] = max_context_tokens
        if where is not None:
            payload["where"] = where

        return self._request("POST", "/query", json=payload, timeout=timeout)

    def list_collections(self) -> list[dict[str, Any]]:
        """
        List all collections in the vector store.
        """
        data = self._request("GET", "/collections")
        return data.get("collections", [])

    def get_collection(self, name: str) -> dict[str, Any]:
        """
        Get metadata and sample chunks for a vector collection.
        """
        return self._request("GET", f"/collections/{name}")

    def delete_collection(self, name: str) -> dict[str, Any]:
        """
        Purge a vector collection from storage.
        """
        return self._request("DELETE", f"/collections/{name}")

    def get_metrics(self) -> dict[str, Any]:
        """
        Retrieve system metrics summary including latency, token usage, and costs.
        """
        return self._request("GET", "/metrics")

    def get_recent_queries(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Retrieve recent query telemetry and profiling records.
        """
        return self._request("GET", "/metrics/queries", params={"limit": limit})

    def get_repo_audit(self, owner: str, repo: str) -> dict[str, Any]:
        """
        Retrieve ingestion file-filtering transparency audit log for a repository.
        """
        return self._request("GET", f"/repos/{owner}/{repo}/audit")

    def reset_metrics(self) -> dict[str, Any]:
        """
        Reset in-memory query telemetry and performance counters.
        """
        return self._request("DELETE", "/metrics")
