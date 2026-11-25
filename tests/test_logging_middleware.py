"""
Unit tests for structured logging configuration, request correlation tracing,
and RequestTracingMiddleware.
Runs offline without network access or external services.
"""
from __future__ import annotations

import io
import json
import logging
import os
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.observability.logging import (
    DEFAULT_LOG_FORMAT,
    JsonLogFormatter,
    RequestIdFilter,
    get_current_request_id,
    reset_current_request_id,
    set_current_request_id,
    setup_logging,
)
from src.api.middleware import (
    PROCESS_TIME_HEADER,
    REQUEST_ID_HEADER,
    RequestTracingMiddleware,
)
from src.api.app import create_app
from src.ui.client import ApiClient, ApiClientError


def test_contextvars_request_id_lifecycle():
    """Verify request ID context variable getter, setter, and reset."""
    assert get_current_request_id() == ""

    token = set_current_request_id("req-trace-123")
    assert get_current_request_id() == "req-trace-123"

    reset_current_request_id(token)
    assert get_current_request_id() == ""


def test_request_id_filter():
    """Verify RequestIdFilter populates record.request_id appropriately."""
    log_filter = RequestIdFilter()
    record = logging.LogRecord("test_logger", logging.INFO, "file.py", 42, "msg", (), None)

    # Without active request ID
    log_filter.filter(record)
    assert record.request_id == ""
    assert record.request_id_raw == ""

    # With active request ID
    token = set_current_request_id("req-999")
    try:
        log_filter.filter(record)
        assert record.request_id == "[req-999] "
        assert record.request_id_raw == "req-999"
    finally:
        reset_current_request_id(token)


def test_json_log_formatter():
    """Verify JsonLogFormatter outputs well-formed JSON with contextual fields."""
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="test_service",
        level=logging.ERROR,
        pathname="service.py",
        lineno=100,
        msg="Error processing item %s",
        args=("item_1",),
        exc_info=None,
    )
    record.request_id_raw = "req-json-abc"

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "ERROR"
    assert data["logger"] == "test_service"
    assert data["message"] == "Error processing item item_1"
    assert data["request_id"] == "req-json-abc"
    assert "timestamp" in data


def test_json_log_formatter_with_exception():
    """Verify JsonLogFormatter includes formatted stack trace when exc_info is present."""
    formatter = JsonLogFormatter()
    try:
        raise ValueError("test exception for logging")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_service",
        level=logging.CRITICAL,
        pathname="service.py",
        lineno=50,
        msg="Fatal failure",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert "exception" in data
    assert "ValueError: test exception for logging" in data["exception"]


def test_setup_logging_text_and_log_level():
    """Verify setup_logging configures root logger handlers and levels."""
    setup_logging(level="DEBUG", log_format="text", force=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG

    managed_handlers = [h for h in root.handlers if getattr(h, "_coderag_managed", False)]
    assert len(managed_handlers) == 1
    assert isinstance(managed_handlers[0].formatter, logging.Formatter)

    # Test re-running setup_logging with JSON
    setup_logging(level="WARNING", log_format="json", force=True)
    assert root.level == logging.WARNING
    managed_handlers = [h for h in root.handlers if getattr(h, "_coderag_managed", False)]
    assert len(managed_handlers) == 1
    assert isinstance(managed_handlers[0].formatter, JsonLogFormatter)

    # Reset back to INFO text
    setup_logging(level="INFO", log_format="text", force=True)


def test_middleware_generates_request_id():
    """Verify middleware generates a unique X-Request-ID when none is provided."""
    test_app = FastAPI()
    test_app.add_middleware(RequestTracingMiddleware)

    @test_app.get("/test-endpoint")
    def sample_endpoint(request: Request):
        return {"req_id_from_state": getattr(request.state, "request_id", None)}

    client = TestClient(test_app)
    resp = client.get("/test-endpoint")

    assert resp.status_code == 200
    assert REQUEST_ID_HEADER in resp.headers
    generated_id = resp.headers[REQUEST_ID_HEADER]
    assert len(generated_id) >= 16
    assert resp.json()["req_id_from_state"] == generated_id
    assert PROCESS_TIME_HEADER in resp.headers
    assert resp.headers[PROCESS_TIME_HEADER].endswith("ms")


def test_middleware_preserves_custom_request_id():
    """Verify middleware honors caller-provided X-Request-ID header."""
    test_app = FastAPI()
    test_app.add_middleware(RequestTracingMiddleware)

    @test_app.get("/test-endpoint")
    def sample_endpoint(request: Request):
        return {"req_id_from_context": get_current_request_id()}

    client = TestClient(test_app)
    custom_id = "client-trace-uuid-12345"
    resp = client.get("/test-endpoint", headers={REQUEST_ID_HEADER: custom_id})

    assert resp.status_code == 200
    assert resp.headers[REQUEST_ID_HEADER] == custom_id
    assert resp.json()["req_id_from_context"] == custom_id


def test_middleware_downstream_logging_tagged(caplog):
    """Verify downstream loggers within the request lifecycle automatically capture request ID."""
    test_app = FastAPI()
    test_app.add_middleware(RequestTracingMiddleware)

    app_logger = logging.getLogger("test_subsystem")

    @test_app.get("/do-work")
    def do_work():
        app_logger.info("Processing business logic inside route handler")
        return {"status": "done"}

    client = TestClient(test_app)
    custom_id = "trace-log-test-99"

    with caplog.at_level(logging.INFO):
        resp = client.get("/do-work", headers={REQUEST_ID_HEADER: custom_id})

    assert resp.status_code == 200
    # Context variable was populated during handler execution
    access_logs = [record for record in caplog.records if record.name == "src.api.access"]
    assert any("Incoming request GET /do-work" in r.message for r in access_logs)
    assert any("Completed GET /do-work -> 200" in r.message for r in access_logs)


def test_middleware_handles_unhandled_exception():
    """Verify middleware intercepts unhandled exceptions, logs them, and returns 500 JSON with request ID."""
    test_app = FastAPI()
    test_app.add_middleware(RequestTracingMiddleware)

    @test_app.get("/crash")
    def crash():
        raise RuntimeError("Unexpected pipeline crash")

    client = TestClient(test_app, raise_server_exceptions=False)
    custom_id = "crash-req-001"
    resp = client.get("/crash", headers={REQUEST_ID_HEADER: custom_id})

    assert resp.status_code == 500
    assert resp.headers[REQUEST_ID_HEADER] == custom_id
    assert PROCESS_TIME_HEADER in resp.headers

    data = resp.json()
    assert data["detail"] == "Internal server error"
    assert data["request_id"] == custom_id


def test_create_app_integration_and_cors_headers():
    """Verify create_app registers middleware and exposes tracking headers in CORS."""
    app = create_app()
    client = TestClient(app)

    # 1. Health check includes headers
    resp = client.get("/health")
    assert resp.status_code == 200
    assert REQUEST_ID_HEADER in resp.headers
    assert PROCESS_TIME_HEADER in resp.headers

    # 2. CORS preflight check exposes headers
    options_headers = {
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "GET",
    }
    options_resp = client.options("/health", headers=options_headers)
    assert options_resp.status_code == 200

    # 3. GET with Origin returns exposed headers header
    cors_resp = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert cors_resp.status_code == 200
    exposed = cors_resp.headers.get("access-control-expose-headers", "")
    assert "X-Request-ID" in exposed
    assert "X-Process-Time" in exposed


def test_ui_api_client_captures_request_id():
    """Verify UI ApiClient stores last_request_id and attaches to ApiClientError."""
    client = ApiClient("http://fake-api:8000")

    fake_resp = MagicMock()
    fake_resp.status_code = 404
    fake_resp.headers = {"X-Request-ID": "backend-trace-555"}
    fake_resp.json.return_value = {"detail": "Repository not found", "request_id": "backend-trace-555"}

    with patch.object(client.session, "request", return_value=fake_resp):
        with pytest.raises(ApiClientError) as exc_info:
            client.get_repo("owner", "nonexistent")

        assert exc_info.value.request_id == "backend-trace-555"
        assert client.last_request_id == "backend-trace-555"


def test_json_log_formatter_includes_extra_and_source_on_warning():
    """Verify JsonLogFormatter serializes extra context fields and source location for >= WARNING."""
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="src.api.access",
        level=logging.WARNING,
        pathname="middleware.py",
        lineno=72,
        msg="Slow request",
        args=(),
        exc_info=None,
    )
    record.http_method = "POST"
    record.status_code = 504
    record.custom_obj = object()

    data = json.loads(formatter.format(record))
    assert data["http_method"] == "POST"
    assert data["status_code"] == 504
    assert "custom_obj" in data
    assert data["source"] == "middleware:72"


def test_setup_logging_unifies_uvicorn_and_dampens_noisy_loggers():
    """Verify setup_logging routes uvicorn loggers to root and sets noisy third-party loggers >= WARNING."""
    uv_error = logging.getLogger("uvicorn.error")
    uv_error.addHandler(logging.StreamHandler())
    uv_error.propagate = False

    setup_logging(level="INFO", log_format="json", force=True)

    try:
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uv_logger = logging.getLogger(name)
            assert uv_logger.handlers == []
            assert uv_logger.propagate is True
            assert uv_logger.level == logging.INFO

        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING
        assert logging.getLogger("urllib3").level >= logging.WARNING
    finally:
        setup_logging(level="INFO", log_format="text", force=True)


def test_middleware_attaches_structured_extra_fields(caplog):
    """Verify RequestTracingMiddleware attaches http_method, http_path, status_code, and duration_ms."""
    test_app = FastAPI()
    test_app.add_middleware(RequestTracingMiddleware)

    @test_app.get("/items")
    def list_items():
        return {"items": []}

    client = TestClient(test_app)
    with caplog.at_level(logging.INFO, logger="src.api.access"):
        resp = client.get("/items", headers={REQUEST_ID_HEADER: "extra-fields-001"})

    assert resp.status_code == 200
    completed = [
        r for r in caplog.records
        if r.name == "src.api.access" and "Completed GET /items" in r.message
    ]
    assert len(completed) == 1
    rec = completed[0]
    assert getattr(rec, "http_method", None) == "GET"
    assert getattr(rec, "http_path", None) == "/items"
    assert getattr(rec, "status_code", None) == 200
    assert isinstance(getattr(rec, "duration_ms", None), float)


def test_app_lifespan_emits_startup_and_shutdown_logs(caplog, monkeypatch):
    """Verify FastAPI lifespan logs structured startup and shutdown events without secrets."""
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-gemini-key")
    monkeypatch.setenv("LOG_FORMAT", "text")
    app = create_app()

    with caplog.at_level(logging.INFO, logger="src.api.app"):
        with TestClient(app) as client:
            resp = client.get("/health/live")
            assert resp.status_code == 200

    messages = [r.message for r in caplog.records if r.name == "src.api.app"]
    assert any("Starting " in m for m in messages)
    assert any("Shutting down " in m for m in messages)
    assert all("super-secret-gemini-key" not in m for m in messages)
    setup_logging(level="INFO", log_format="text", force=True)


def test_api_main_configures_uvicorn_without_duplicate_access_logs():
    """Verify src.api.main passes log_config=None and access_log=False to uvicorn.run."""
    from src.api.main import main as api_main

    with patch("sys.argv", ["src.api.main", "--host", "127.0.0.1", "--port", "9001"]), \
         patch("uvicorn.run") as mock_run:
        api_main()

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9001
    assert kwargs["log_config"] is None
    assert kwargs["access_log"] is False
    setup_logging(level="INFO", log_format="text", force=True)
