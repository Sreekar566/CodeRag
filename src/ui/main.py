"""
CLI runner for launching the Streamlit Frontend UI.
Usage:
    python -m src.ui.main --host 0.0.0.0 --port 8501 --api-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Codebase Intelligence Platform - Frontend UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8501, help="Port to listen on (default: 8501)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend API server base URL")
    parser.add_argument("--headless", action="store_true", default=True, help="Run Streamlit in headless mode")
    return parser.parse_args(argv)


def get_app_path() -> Path:
    """Return absolute path to app.py."""
    return Path(__file__).parent / "app.py"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Propagate backend API URL to environment for Streamlit process
    os.environ["API_BASE_URL"] = args.api_url

    app_path = get_app_path()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--server.headless",
        "true" if args.headless else "false",
    ]

    print(f"Launching Streamlit UI at http://{args.host}:{args.port} (connected to {args.api_url})...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped.")


if __name__ == "__main__":
    main()
