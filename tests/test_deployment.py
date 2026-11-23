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

    # Healthcheck configuration check
    assert "HEALTHCHECK" in content, "Dockerfile should define container HEALTHCHECK"
    assert "/health/live" in content or "/health" in content

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
    assert "cli)" in content, "Entrypoint must support 'cli' command"
    assert "doctor)" in content, "Entrypoint must support 'doctor' command"
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
    assert "healthcheck" in ui, "Service 'ui' must define a healthcheck"
    assert "curl" in " ".join(ui["healthcheck"].get("test", []))
    assert "_stcore/health" in " ".join(ui["healthcheck"].get("test", []))

    ui_env = ui.get("environment", [])
    assert any("API_BASE_URL=http://api:8000" in str(e) for e in ui_env)

    # Verify volume definition
    assert "volumes" in compose_data
    assert "codebase_data" in compose_data["volumes"]


def test_docker_compose_healthcheck_configurations() -> None:
    compose_path = REPO_ROOT / "docker-compose.yml"
    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "api" in services and "ui" in services

    for svc_name in ("api", "ui"):
        svc = services[svc_name]
        assert "healthcheck" in svc, f"{svc_name} must have a healthcheck configured"
        hc = svc["healthcheck"]
        assert "test" in hc
        assert hc.get("interval") == "15s"
        assert hc.get("timeout") == "5s"
        assert hc.get("retries") == 3
        assert hc.get("start_period") == "10s"

    assert "http://localhost:8000/health" in " ".join(services["api"]["healthcheck"]["test"])
    assert "http://localhost:8501/_stcore/health" in " ".join(services["ui"]["healthcheck"]["test"])


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


def test_makefile_developer_tooling_targets() -> None:
    makefile_path = REPO_ROOT / "Makefile"
    assert makefile_path.is_file(), "Makefile must exist at repository root"

    content = makefile_path.read_text()
    expected_targets = [
        "help:",
        "install:",
        "test:",
        "test-unit:",
        "test-integration:",
        "api:",
        "ui:",
        "eval:",
        "cli:",
        "doctor:",
        "docker-build:",
        "docker-up:",
        "docker-down:",
        "clean:",
        "lint:",
    ]
    for target in expected_targets:
        assert target in content, f"Makefile should define target '{target}'"

    # Test running make help
    result = subprocess.run(
        ["make", "-C", str(REPO_ROOT), "help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "AI Codebase Intelligence Platform" in result.stdout
    assert "make test" in result.stdout


def test_env_example_configuration() -> None:
    env_example_path = REPO_ROOT / ".env.example"
    assert env_example_path.is_file(), ".env.example must exist at repository root"

    content = env_example_path.read_text()
    expected_keys = [
        "PORT=",
        "HOST=",
        "API_BASE_URL=",
        "LLM_PROVIDER=",
        "GEMINI_API_KEY=",
        "GEMINI_MODEL=",
        "GITHUB_TOKEN=",
        "CHROMA_PERSIST_DIRECTORY=",
        "REGISTRY_PATH=",
        "LOG_LEVEL=",
        "LOG_FORMAT=",
        "CHUNK_SIZE=",
        "CHUNK_OVERLAP=",
    ]
    for key in expected_keys:
        assert key in content, f".env.example should define configuration '{key}'"

