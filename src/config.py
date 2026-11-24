"""
Centralized, validated configuration settings for the AI Codebase Intelligence Platform.
Powered by Pydantic Settings (BaseSettings) and dotenv support.
"""
from __future__ import annotations

import os
import re
from typing import Any
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_LOG_LEVELS: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_LOG_FORMATS: set[str] = {"text", "json"}
VALID_LLM_PROVIDERS: set[str] = {"gemini", "fake"}


class Settings(BaseSettings):
    """
    Application-wide configuration settings with environment variable resolution,
    strict type validation, normalization, and sensible defaults.
    """
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # Platform / API Server
    # -------------------------------------------------------------------------
    app_name: str = Field(
        default="AI Codebase Intelligence Platform",
        description="Application display name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version string",
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("host", "api_host"),
        description="Network host interface to bind API server",
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("port", "api_port"),
        description="Port for API server (1-65535)",
    )
    reload: bool = Field(
        default=False,
        validation_alias=AliasChoices("reload", "api_reload"),
        description="Enable automatic code reloading in development",
    )
    workers: int = Field(
        default=1,
        validation_alias=AliasChoices("workers", "api_workers"),
        description="Number of uvicorn worker processes (>= 1)",
    )
    api_base_url: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("api_base_url"),
        description="Base URL for the backend API (used by UI client)",
    )

    # -------------------------------------------------------------------------
    # Storage & Registry
    # -------------------------------------------------------------------------
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        validation_alias=AliasChoices("chroma_persist_directory"),
        description="Directory for local persistent Chroma vector database",
    )
    registry_path: str = Field(
        default="./data/registry.json",
        validation_alias=AliasChoices("registry_path"),
        description="File path for JSON repository registry metadata",
    )
    default_collection_name: str = Field(
        default="codebase",
        validation_alias=AliasChoices("default_collection_name"),
        description="Default vector store collection name",
    )
    storage_batch_size: int = Field(
        default=500,
        validation_alias=AliasChoices("storage_batch_size"),
        description="Batch size for writing chunk embeddings to storage",
    )

    # -------------------------------------------------------------------------
    # Embedding Model
    # -------------------------------------------------------------------------
    embedding_model: str = Field(
        default="flax-sentence-embeddings/st-codesearch-distilroberta-base",
        validation_alias=AliasChoices("embedding_model"),
        description="SentenceTransformer model identifier for code embedding",
    )

    # -------------------------------------------------------------------------
    # LLM Provider
    # -------------------------------------------------------------------------
    llm_provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("llm_provider"),
        description="LLM provider name: 'gemini' or 'fake'. If None, auto-detected from GEMINI_API_KEY.",
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("gemini_api_key"),
        description="Google Gemini API key for real LLM completions",
    )
    gemini_model: str = Field(
        default="gemini-3.8-flash",
        validation_alias=AliasChoices("gemini_model"),
        description="Default Gemini model identifier",
    )

    # -------------------------------------------------------------------------
    # GitHub Integration
    # -------------------------------------------------------------------------
    github_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("github_token"),
        description="GitHub personal access token for higher rate limits",
    )
    github_api_base: str = Field(
        default="https://api.github.com",
        validation_alias=AliasChoices("github_api_base"),
        description="Base URL for GitHub REST API",
    )

    # -------------------------------------------------------------------------
    # Logging & Observability
    # -------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("log_level"),
        description="Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')",
    )
    log_format: str = Field(
        default="text",
        validation_alias=AliasChoices("log_format"),
        description="Logging format style ('text' or 'json')",
    )

    # -------------------------------------------------------------------------
    # Chunking & RAG Parameters
    # -------------------------------------------------------------------------
    chunk_size: int = Field(
        default=400,
        validation_alias=AliasChoices("chunk_size"),
        description="Sliding window chunk size in approximate words/tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        validation_alias=AliasChoices("chunk_overlap"),
        description="Overlap between consecutive chunks in approximate words/tokens",
    )
    boundary_search_window: int = Field(
        default=5,
        validation_alias=AliasChoices("boundary_search_window"),
        description="Number of lines to search outward to snap chunks to blank lines",
    )
    max_context_tokens: int = Field(
        default=2048,
        validation_alias=AliasChoices("max_context_tokens"),
        description="Maximum prompt token budget for retrieved codebase context",
    )

    # -------------------------------------------------------------------------
    # Field Validators
    # -------------------------------------------------------------------------
    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("host cannot be empty or whitespace")
        return cleaned

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Workers must be at least 1, got {v}")
        return v

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, v: str) -> str:
        cleaned = v.strip()
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            raise ValueError(
                f"Invalid api_base_url '{v}'. Must start with 'http://' or 'https://'"
            )
        return cleaned

    @field_validator("chroma_persist_directory")
    @classmethod
    def validate_chroma_persist_directory(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("chroma_persist_directory cannot be empty or whitespace")
        return cleaned

    @field_validator("registry_path")
    @classmethod
    def validate_registry_path(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("registry_path cannot be empty or whitespace")
        if not cleaned.lower().endswith(".json"):
            raise ValueError(
                f"registry_path must have a .json file extension, got '{v}'"
            )
        return cleaned

    @field_validator("default_collection_name")
    @classmethod
    def validate_default_collection_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("default_collection_name cannot be empty or whitespace")
        if not (3 <= len(cleaned) <= 63):
            raise ValueError(
                f"default_collection_name length must be between 3 and 63 characters, got {len(cleaned)}"
            )
        if not re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
            raise ValueError(
                f"default_collection_name '{v}' contains invalid characters. Must be alphanumeric, underscores, or hyphens."
            )
        return cleaned

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = v.lower().strip()
        if not normalized:
            return None
        if normalized not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: '{v}'. Supported providers are: {', '.join(sorted(VALID_LLM_PROVIDERS))}"
            )
        return normalized

    @field_validator("gemini_api_key", "github_token")
    @classmethod
    def validate_credentials(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("embedding_model cannot be empty or whitespace")
        return cleaned

    @field_validator("gemini_model")
    @classmethod
    def validate_gemini_model(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("gemini_model cannot be empty or whitespace")
        return cleaned

    @field_validator("github_api_base")
    @classmethod
    def validate_github_api_base(cls, v: str) -> str:
        cleaned = v.strip().rstrip("/")
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            raise ValueError(
                f"Invalid github_api_base '{v}'. Must start with 'http://' or 'https://'"
            )
        return cleaned

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        normalized = v.upper().strip()
        if normalized not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level '{v}'. Must be one of: {', '.join(sorted(VALID_LOG_LEVELS))}"
            )
        return normalized

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        normalized = v.lower().strip()
        if normalized not in VALID_LOG_FORMATS:
            raise ValueError(
                f"Invalid log_format '{v}'. Must be one of: {', '.join(sorted(VALID_LOG_FORMATS))}"
            )
        return normalized

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 50:
            raise ValueError(f"chunk_size must be >= 50, got {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {v}")
        return v

    @field_validator("boundary_search_window")
    @classmethod
    def validate_boundary_search_window(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"boundary_search_window must be >= 0, got {v}")
        return v

    @field_validator("max_context_tokens")
    @classmethod
    def validate_max_context_tokens(cls, v: int) -> int:
        if v < 64:
            raise ValueError(f"max_context_tokens must be >= 64, got {v}")
        return v

    @field_validator("storage_batch_size")
    @classmethod
    def validate_storage_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"storage_batch_size must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def validate_overlap_less_than_chunk_size(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        if self.chunk_size > self.max_context_tokens:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) cannot exceed max_context_tokens ({self.max_context_tokens})"
            )
        return self

    # -------------------------------------------------------------------------
    # Helper Properties and Methods
    # -------------------------------------------------------------------------
    @property
    def effective_llm_provider(self) -> str:
        """
        Resolve the effective LLM provider:
        1. Explicit llm_provider if configured ('gemini' or 'fake')
        2. 'gemini' if gemini_api_key is set
        3. 'fake' as offline/testing default fallback
        """
        if self.llm_provider and self.llm_provider.strip():
            return self.llm_provider.lower().strip()
        if self.gemini_api_key and self.gemini_api_key.strip():
            return "gemini"
        return "fake"

    @property
    def is_gemini_available(self) -> bool:
        """Check if Gemini credentials are configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def is_github_token_available(self) -> bool:
        """Check if a GitHub personal access token is configured."""
        return bool(self.github_token and self.github_token.strip())

    @property
    def is_local_storage(self) -> bool:
        """Check if Chroma is configured for local persistent disk storage."""
        return bool(self.chroma_persist_directory and self.chroma_persist_directory.strip())

    def to_masked_dict(self) -> dict[str, Any]:
        """
        Export configuration as a dictionary with sensitive credentials masked
        for safe logging, inspection, and telemetry reporting.
        """
        data = self.model_dump()
        for secret_key in ("gemini_api_key", "github_token"):
            val = data.get(secret_key)
            if val and isinstance(val, str):
                if len(val) > 6:
                    data[secret_key] = f"{val[:3]}...{val[-2:]}"
                else:
                    data[secret_key] = "***"
        return data


def get_settings() -> Settings:
    """
    Construct a Settings instance from the current environment and .env file.
    """
    return Settings()
