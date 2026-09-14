"""Locate the sibling Oshara project and its private Python environment."""
from __future__ import annotations

import os
import sys
from pathlib import Path


_VOICE_STUDIO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OSHARA_DIR = _VOICE_STUDIO_ROOT.parent / "oshara_xtts_v2"
OSHARA_SIDECAR_SCRIPT = Path(__file__).with_name("main.py")


def resolve_oshara_dir() -> Path:
    return Path(
        os.environ.get("OMNIVOICE_OSHARA_XTTS_DIR", str(_DEFAULT_OSHARA_DIR))
    ).expanduser().resolve()


def resolve_oshara_venv() -> Path:
    root = resolve_oshara_dir()
    name = "python.exe" if sys.platform == "win32" else "python"
    return root / ".venv" / "Scripts" / name if sys.platform == "win32" else root / ".venv" / "bin" / name


def is_oshara_installed() -> bool:
    root = resolve_oshara_dir()
    return (root / "app.py").is_file() and resolve_oshara_venv().is_file()


__all__ = [
    "OSHARA_SIDECAR_SCRIPT",
    "is_oshara_installed",
    "resolve_oshara_dir",
    "resolve_oshara_venv",
]