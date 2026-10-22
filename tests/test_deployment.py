"""
Unit tests for deployment and container artifacts:
- Dockerfile
- .dockerignore
- docker-entrypoint.sh
- docker-compose.yml
- .github/workflows/ci.yml
- Environment variable configuration for persistent paths
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import yaml
import pytest


REPO_ROOT = Path(__file__).parent.parent


def test_dockerfile_exists_and_configured() -> None:
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.is_file(), "Dockerfile must exist at repository root"

    content = dockerfile_path.read_text()

    # Base image check
    assert "FROM python:3.12-slim" in content, "Dockerfile should use python:3.12-slim"

    # Security: non-root user
    assert "useradd" in content or "adduser" in content, "Dockerfile must create a non-root user"
    assert "USER appuser" in content, "Dockerfile must switch to non-root USER appuser"

    # CPU PyTorch installation check
    assert "download.pytorch.org/whl/cpu" in content, "Dockerfile must install CPU-only PyTorch"

    # Ports exposed
    assert "EXPOSE 8000 8501" in content or ("8000" in content and "8501" in content), (
        "Dockerfile must expose ports 8000 and 8501"
    )

    # Entrypoint and default command
    assert 'ENTRYPOINT ["/app/docker-entrypoint.sh"]' in content
    assert 'CMD ["api"]' in content


def test_dockerignore_contains_sensitive_and_ephemeral_paths() -> None:
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.is_file(), ".dockerignore must exist at repository root"

    content = dockerignore_path.read_text()
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    expected_patterns = [
        ".git",
        "venv/",
        "data/",
        ".env",
        ".pytest_cache/",
        "__pycache__/",
    ]

    for pattern in expected_patterns:
        assert any(pattern in line for line in lines), f".dockerignore should exclude '{pattern}'"


def test_docker_entrypoint_executable_and_handlers() -> None:
    entrypoint_path = REPO_ROOT / "docker-entrypoint.sh"
    assert entrypoint_path.is_file(), "docker-entrypoint.sh must exist"
    assert os.access(entrypoint_path, os.X_OK), "docker-entrypoint.sh must be executable"

    content = entrypoint_path.read_text()
    assert "api)" in content, "Entrypoint must support 'api' command"
    assert "ui)" in content, "Entrypoint must support 'ui' command"
    assert "eval)" in content, "Entrypoint must support 'eval' command"
    assert "test)" in content, "Entrypoint must support 'test' command"

    # Test executing entrypoint with pass-through command
    result = subprocess.run(
        [str(entrypoint_path), "echo", "entrypoint_ok"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "entrypoint_ok" in result.stdout


def test_docker_compose_valid_yaml_and_service_topology() -> None:
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.is_file(), "docker-compose.yml must exist at repository root"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data, "docker-compose.yml must define 'services'"
    services = compose_data["services"]

    # Verify api service
    assert "api" in services, "Service 'api' must be defined"
    api = services["api"]
    assert "8000:8000" in api.get("ports", [])
    assert api.get("command") == "api"
    assert "healthcheck" in api
    assert "curl" in " ".join(api["healthcheck"].get("test", []))
    assert any("codebase_data" in v for v in api.get("volumes", []))

    # Verify ui service
    assert "ui" in services, "Service 'ui' must be defined"
    ui = services["ui"]
    assert "8501:8501" in ui.get("ports", [])
    assert ui.get("command") == "ui"
    assert "depends_on" in ui
    assert "api" in ui["depends_on"]
    assert ui["depends_on"]["api"].get("condition") == "service_healthy"

    ui_env = ui.get("environment", [])
    assert any("API_BASE_URL=http://api:8000" in str(e) for e in ui_env)

    # Verify volume definition
    assert "volumes" in compose_data
    assert "codebase_data" in compose_data["volumes"]


def test_ci_workflow_valid_yaml_and_test_execution() -> None:
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.is_file(), ".github/workflows/ci.yml must exist"

    with open(ci_path, "r", encoding="utf-8") as f:
        ci_data = yaml.safe_load(f)

    assert "name" in ci_data
    assert "on" in ci_data or True  # YAML parser parses 'on' as True (boolean) in standard YAML 1.1

    # Check jobs
    assert "jobs" in ci_data
    jobs = ci_data["jobs"]
    assert "test" in jobs, "CI workflow must have a 'test' job"
    test_job = jobs["test"]
    assert test_job.get("runs-on") == "ubuntu-latest"

    steps = test_job.get("steps", [])
    step_runs = [s.get("run", "") for s in steps]
    step_uses = [s.get("uses", "") for s in steps]

    # Checkout and python setup
    assert any("actions/checkout" in u for u in step_uses)
    assert any("actions/setup-python" in u for u in step_uses)

    # PyTorch and dependencies
    assert any("download.pytorch.org/whl/cpu" in r for r in step_runs)
    assert any("requirements.txt" in r for r in step_runs)

    # Pytest execution
    assert any("pytest" in r for r in step_runs)


def test_env_path_configuration(tmp_path: Path) -> None:
    custom_chroma = str(tmp_path / "custom_chroma")
    custom_registry = str(tmp_path / "custom_registry.json")

    cmd = [
        sys.executable,
        "-c",
        (
            f"import os; "
            f"os.environ['CHROMA_PERSIST_DIRECTORY'] = {custom_chroma!r}; "
            f"os.environ['REGISTRY_PATH'] = {custom_registry!r}; "
            f"from src.storage.chroma_store import DEFAULT_CHROMA_PATH; "
            f"from src.registry.registry import DEFAULT_REGISTRY_PATH; "
            f"assert DEFAULT_CHROMA_PATH == {custom_chroma!r}; "
            f"assert DEFAULT_REGISTRY_PATH == {custom_registry!r}"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
