"""AI4Bharat IndicConformer Nepali ASR subprocess backend."""
from __future__ import annotations

from typing import TYPE_CHECKING

from services.subprocess_asr import SubprocessASRBackend

if TYPE_CHECKING:
    from pathlib import Path


class IndicConformerNepaliBackend(SubprocessASRBackend):
    """Nepali IndicConformer hybrid CTC/RNNT ASR model."""

    id = "indicconformer-nepali"
    display_name = "IndicConformer (AI4Bharat Nepali)"
    gpu_compat = ("cuda", "cpu")

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        from engines.indicconformer_nepali.bootstrap import (
            INDICCONFORMER_SIDECAR_SCRIPT,
            is_indicconformer_installed,
        )

        if not is_indicconformer_installed():
            return False, (
                "IndicConformer Nepali environment not found. Set "
                "OMNIVOICE_INDICCONFORMER_ASR_DIR to its dedicated environment "
                "directory and install AI4Bharat NeMo there."
            )
        if not INDICCONFORMER_SIDECAR_SCRIPT.exists():
            return False, "IndicConformer Nepali sidecar script is missing."
        return True, "ready"

    def _spawn(self) -> None:
        """Hand the sidecar a decoder before it starts.

        The sidecar's own environment carries a plain libsndfile build, which
        reads WAV/FLAC/OGG but not WebM/Opus, MP4/AAC or the rest of what a
        user drops on the app — and dictation's browser capture is WebM/Opus.
        The parent already resolves an ffmpeg (bundled static binary, user
        override, or system), so publish that choice into the child's env
        rather than making the engine venv carry its own codec stack.
        """
        import os

        if not os.environ.get("FFMPEG_PATH"):
            try:
                from services.ffmpeg_utils import find_ffmpeg
                resolved = find_ffmpeg()
                if resolved:
                    os.environ["FFMPEG_PATH"] = str(resolved)
            except Exception:  # noqa: BLE001 — the sidecar degrades to libsndfile
                pass
        super()._spawn()

    @classmethod
    def venv_python(cls) -> "Path":
        from engines.indicconformer_nepali.bootstrap import resolve_venv_python
        return resolve_venv_python()

    @classmethod
    def sidecar_script(cls) -> "Path":
        from engines.indicconformer_nepali.bootstrap import INDICCONFORMER_SIDECAR_SCRIPT
        return INDICCONFORMER_SIDECAR_SCRIPT


__all__ = ["IndicConformerNepaliBackend"]