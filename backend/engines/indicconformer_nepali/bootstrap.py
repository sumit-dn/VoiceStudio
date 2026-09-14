"""Locate the dedicated IndicConformer Nepali Python environment."""
from __future__ import annotations

import os
import sys
from pathlib import Path


_DEFAULT_ENGINE_DIR = Path.home() / ".omnivoice" / "engines" / "indicconformer-nepali"
INDICCONFORMER_SIDECAR_SCRIPT = Path(__file__).with_name("main.py")


def resolve_engine_dir() -> Path:
    return Path(
        os.environ.get("OMNIVOICE_INDICCONFORMER_ASR_DIR", str(_DEFAULT_ENGINE_DIR))
    ).expanduser().resolve()


def resolve_venv_python() -> Path:
    root = resolve_engine_dir()
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def is_indicconformer_installed() -> bool:
    return resolve_venv_python().is_file()


__all__ = [
    "INDICCONFORMER_SIDECAR_SCRIPT",
    "is_indicconformer_installed",
    "resolve_engine_dir",
    "resolve_venv_python",
]