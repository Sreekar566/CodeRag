"""
CLI runner for launching the FastAPI backend server.
Usage:
    python -m src.api.main --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import argparse
import uvicorn

from src.config import get_settings
from src.observability.logging import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging(force=True)
    parser = argparse.ArgumentParser(description="AI Codebase Intelligence Platform API Server")
    parser.add_argument("--host", default=settings.host, help=f"Host interface to bind to (default: {settings.host})")
    parser.add_argument("--port", type=int, default=settings.port, help=f"Port to listen on (default: {settings.port})")
    parser.add_argument("--reload", action="store_true", default=settings.reload, help="Enable automatic code reloading")
    parser.add_argument("--workers", type=int, default=settings.workers, help=f"Number of worker processes (default: {settings.workers})")

    args = parser.parse_args()

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_config=None,
        access_log=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
