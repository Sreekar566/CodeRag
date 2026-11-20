"""
Developer CLI package for the AI Codebase Intelligence Platform.
"""
from __future__ import annotations

__all__ = ["main", "build_parser"]


def __getattr__(name: str):
    if name in ("main", "build_parser"):
        from src.cli.main import build_parser, main

        globals()["main"] = main
        globals()["build_parser"] = build_parser
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
