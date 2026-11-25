"""
HTTP middleware for request correlation, latency profiling, and access logging.
"""
from __future__ import annotations

import logging
import time
from typing import Callable
import uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.observability.logging import (
    reset_current_request_id,
    set_current_request_id,
)

logger = logging.getLogger("src.api.access")

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time"


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that manages request correlation IDs, monotonic process timing,
    contextual request-scoped logging, and graceful unhandled exception handling.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        raw_req_id = request.headers.get(REQUEST_ID_HEADER)
        if raw_req_id and raw_req_id.strip():
            request_id = raw_req_id.strip()
        else:
            request_id = uuid.uuid4().hex

        # Store on request state for dependency or endpoint access
        request.state.request_id = request_id

        # Bind contextvar for downstream log records
        token = set_current_request_id(request_id)
        start_time = time.perf_counter()

        client_host = request.client.host if request.client else "unknown"
        request_extra = {
            "http_method": request.method,
            "http_path": request.url.path,
            "client_ip": client_host,
        }
        logger.info(
            f"Incoming request {request.method} {request.url.path} from {client_host}",
            extra=request_extra,
        )

        try:
            try:
                response = await call_next(request)
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                logger.error(
                    f"Unhandled exception during {request.method} {request.url.path} "
                    f"after {duration_ms:.2f}ms: {exc}",
                    exc_info=True,
                    extra={
                        **request_extra,
                        "status_code": 500,
                        "duration_ms": round(duration_ms, 2),
                    },
                )
                response = JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Internal server error",
                        "request_id": request_id,
                    },
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.2f}ms"

            logger.info(
                f"Completed {request.method} {request.url.path} -> {response.status_code} "
                f"in {duration_ms:.2f}ms",
                extra={
                    **request_extra,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return response
        finally:
            reset_current_request_id(token)
