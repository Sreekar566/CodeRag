"""
Unit tests for centralized configuration management and Pydantic Settings validation.
Tests verify default values, type coercion, environment overrides, and validator constraints.
"""
from __future__ import annotations

import os
import pytest
from pydantic import ValidationError
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from src.config import Settings, get_settings


def test_default_settings():
    """Verify default settings values and types."""
    settings = Settings()
    assert settings.app_name == "AI Codebase Intelligence Platform"
    assert settings.app_version == "0.1.0"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.reload is False
    assert settings.workers == 1
    assert settings.api_base_url == "http://localhost:8000"

    assert settings.chroma_persist_directory == "./data/chroma"
    assert settings.registry_path == "./data/registry.json"
    assert settings.default_collection_name == "codebase"
    assert settings.storage_batch_size == 500

    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.github_api_base == "https://api.github.com"

    assert settings.log_level == "INFO"
    assert settings.log_format == "text"

    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 50
    assert settings.boundary_search_window == 5
    assert settings.max_context_tokens == 2048


def test_env_var_overrides(monkeypatch):
    """Verify settings correctly parse and normalize environment variables."""
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("WORKERS", "4")
    monkeypatch.setenv("RELOAD", "true")
    monkeypatch.setenv("API_BASE_URL", "http://myapi:9090/")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/custom/chroma")
    monkeypatch.setenv("REGISTRY_PATH", "/custom/registry.json")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "JSON")
    monkeypatch.setenv("LLM_PROVIDER", "FAKE")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secrettoken")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    settings = get_settings()

    assert settings.port == 9090
    assert settings.host == "127.0.0.1"
    assert settings.workers == 4
    assert settings.reload is True
    assert settings.api_base_url == "http://myapi:9090/"
    assert settings.chroma_persist_directory == "/custom/chroma"
    assert settings.registry_path == "/custom/registry.json"
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "json"
    assert settings.llm_provider == "fake"
    assert settings.gemini_api_key == "test-key-123"
    assert settings.github_token == "ghp_secrettoken"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 100


def test_invalid_port_validation():
    """Verify port bounds validation (1-65535)."""
    with pytest.raises(ValidationError) as exc:
        Settings(port=0)
    assert "Port must be between 1 and 65535" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        Settings(port=70000)
    assert "Port must be between 1 and 65535" in str(exc.value)


def test_invalid_workers_validation():
    """Verify workers must be >= 1."""
    with pytest.raises(ValidationError) as exc:
        Settings(workers=0)
    assert "Workers must be at least 1" in str(exc.value)


def test_invalid_log_level_validation():
    """Verify log_level validation and error reporting."""
    with pytest.raises(ValidationError) as exc:
        Settings(log_level="VERBOSE")
    assert "Invalid log_level 'VERBOSE'" in str(exc.value)


def test_invalid_log_format_validation():
    """Verify log_format must be 'text' or 'json'."""
    with pytest.raises(ValidationError) as exc:
        Settings(log_format="yaml")
    assert "Invalid log_format 'yaml'" in str(exc.value)


def test_invalid_llm_provider_validation():
    """Verify unsupported LLM provider raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        Settings(llm_provider="openai")
    assert "Unsupported LLM provider: 'openai'" in str(exc.value)


def test_chunk_constraints_validation():
    """Verify chunk size and overlap constraints."""
    with pytest.raises(ValidationError) as exc:
        Settings(chunk_size=30)
    assert "chunk_size must be >= 50" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        Settings(chunk_overlap=-5)
    assert "chunk_overlap must be >= 0" in str(exc.value)

    # Overlap >= size
    with pytest.raises(ValidationError) as exc:
        Settings(chunk_size=400, chunk_overlap=400)
    assert "must be less than chunk_size" in str(exc.value)


def test_context_budget_validation():
    """Verify max_context_tokens constraint (>= 64)."""
    with pytest.raises(ValidationError) as exc:
        Settings(max_context_tokens=32)
    assert "max_context_tokens must be >= 64" in str(exc.value)


def test_storage_batch_size_validation():
    """Verify storage_batch_size must be >= 1."""
    with pytest.raises(ValidationError) as exc:
        Settings(storage_batch_size=0)
    assert "storage_batch_size must be >= 1" in str(exc.value)


def test_effective_llm_provider_resolution():
    """Verify resolution hierarchy: explicit provider > gemini_key > fake fallback."""
    # 1. Explicit fake provider
    s1 = Settings(llm_provider="fake", gemini_api_key="key")
    assert s1.effective_llm_provider == "fake"

    # 2. Explicit gemini provider
    s2 = Settings(llm_provider="gemini", gemini_api_key="key")
    assert s2.effective_llm_provider == "gemini"

    # 3. Inferred gemini from API key
    s3 = Settings(llm_provider=None, gemini_api_key="some-key")
    assert s3.effective_llm_provider == "gemini"

    # 4. Default fallback to fake when no key and no provider
    s4 = Settings(llm_provider=None, gemini_api_key=None)
    assert s4.effective_llm_provider == "fake"


def test_helper_properties():
    """Verify boolean availability helper properties."""
    s_empty = Settings(gemini_api_key=None, github_token=None)
    assert s_empty.is_gemini_available is False
    assert s_empty.is_github_token_available is False

    s_configured = Settings(gemini_api_key="valid_key", github_token="valid_token")
    assert s_configured.is_gemini_available is True
    assert s_configured.is_github_token_available is True


def test_fastapi_dependency_injection():
    """Verify get_settings acts as a valid FastAPI dependency."""
    app = FastAPI()

    @app.get("/config")
    def read_config(settings: Settings = Depends(get_settings)):
        return {"app": settings.app_name, "version": settings.app_version}

    client = TestClient(app)
    resp = client.get("/config")
    assert resp.status_code == 200
    assert resp.json()["version"] == "0.1.0"
