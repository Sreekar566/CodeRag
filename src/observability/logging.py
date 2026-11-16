"""
Application-wide logging configuration and contextual request ID propagation.
Provides structured text and JSON formatting, contextvars-based request tracing,
and environment variable configuration.
"""
from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(request_id)s%(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_current_request_id() -> str:
    """Retrieve the active request correlation ID in the current execution context."""
    return request_id_ctx.get()


def set_current_request_id(request_id: str) -> contextvars.Token[str]:
    """Set the active request correlation ID for the current context."""
    return request_id_ctx.set(request_id)


def reset_current_request_id(token: contextvars.Token[str]) -> None:
    """Reset the active request correlation ID to its prior context state."""
    request_id_ctx.reset(token)


class RequestIdFilter(logging.Filter):
    """
    Attaches the current request correlation ID from contextvars
    to every LogRecord for unified logging formatters.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        req_id = get_current_request_id()
        record.request_id = f"[{req_id}] " if req_id else ""
        record.request_id_raw = req_id
        return True


class JsonLogFormatter(logging.Formatter):
    """
    Structured JSON log formatter for cloud logging and container observability.
    Serializes timestamp, log level, logger name, message, request ID, and stack traces.
    """

    def format(self, record: logging.LogRecord) -> str:
        req_id = getattr(record, "request_id_raw", "") or get_current_request_id()
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if req_id:
            payload["request_id"] = req_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_LOGGING_CONFIGURED = False


def setup_logging(
    level: str | int | None = None,
    log_format: str | None = None,
    force: bool = False,
) -> None:
    """
    Configure application root and package loggers with standard formatting,
    request ID correlation filters, and log level resolution from environment.

    Args:
        level: Explicit log level (e.g. 'DEBUG', 'INFO') or int. Defaults to LOG_LEVEL env var or 'INFO'.
        log_format: Format style: 'text' or 'json'. Defaults to LOG_FORMAT env var or 'text'.
        force: If True, reconfigure handlers even if already configured.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    # Determine log level
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper().strip()
        resolved_level = getattr(logging, level_str, logging.INFO)
    elif isinstance(level, str):
        resolved_level = getattr(logging, level.upper().strip(), logging.INFO)
    else:
        resolved_level = level

    # Determine format style
    if log_format is None:
        format_style = os.getenv("LOG_FORMAT", "text").lower().strip()
    else:
        format_style = log_format.lower().strip()

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    # Remove any existing StreamHandlers attached during previous setup
    for handler in list(root_logger.handlers):
        if getattr(handler, "_coderag_managed", False):
            root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(resolved_level)
    stream_handler.addFilter(RequestIdFilter())
    setattr(stream_handler, "_coderag_managed", True)

    if format_style == "json":
        stream_handler.setFormatter(JsonLogFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )

    root_logger.addHandler(stream_handler)
    _LOGGING_CONFIGURED = True
