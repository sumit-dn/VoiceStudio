"""Oshara Nepali XTTS-v2 subprocess backend.

Wraps ``Oshara/xtts-v2-nepali`` (Coqui XTTS v2 fine-tuned to add Nepali,
24 kHz, zero-shot cloning from a reference clip).

NOT an upstream candidate as-is: the weights inherit the Coqui Public Model
License (https://coqui.ai/cpml), which is non-commercial, so this fails bar #2
of docs/engine-acceptance.md ("Licence clean for commercial use. Model weights
*and* code."). The only approved exception there is audiocpp/Breeze-TTS-2,
granted explicitly by the owner. This engine is a local try-out: it stays
opt-in and discloses the restriction before selection and download.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from services.subprocess_backend import SubprocessBackend

if TYPE_CHECKING:
    import torch


class OsharaXTTSV2Backend(SubprocessBackend):
    """Nepali XTTS-v2 engine from the sibling ``oshara_xtts_v2`` project."""

    id = "oshara-xtts-v2"
    display_name = "Oshara XTTS-v2 (Nepali, CPML non-commercial)"
    _DEFAULT_SAMPLE_RATE = 24000
    # CPU XTTS inference can exceed the generic 60-second subprocess deadline
    # for long Nepali prompts; the sidecar remains alive while this call runs.
    recv_timeout_s = 300.0
    gpu_compat = ("cuda", "cpu")
    supports_cloning = True

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        from engines.oshara_xtts_v2.bootstrap import (
            OSHARA_SIDECAR_SCRIPT,
            is_oshara_installed,
        )

        if not is_oshara_installed():
            return False, (
                "Oshara XTTS-v2 environment not found. Set "
                "OMNIVOICE_OSHARA_XTTS_DIR to the oshara_xtts_v2 project "
                "and run `uv sync` there."
            )
        if not OSHARA_SIDECAR_SCRIPT.exists():
            return False, "Oshara XTTS-v2 sidecar script is missing."
        return True, "ready (non-commercial CPML weights)"

    @classmethod
    def venv_python(cls):
        from engines.oshara_xtts_v2.bootstrap import resolve_oshara_venv
        return resolve_oshara_venv()

    @classmethod
    def sidecar_script(cls):
        from engines.oshara_xtts_v2.bootstrap import OSHARA_SIDECAR_SCRIPT
        return OSHARA_SIDECAR_SCRIPT

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        return ["ne"]

    def generate(self, text: str, **kw) -> "torch.Tensor":
        forwarded = {
            "language": kw.get("language"),
            "ref_audio": kw.get("ref_audio"),
            "temperature": kw.get("temperature", 0.7),
            "length_penalty": kw.get("length_penalty", 1.0),
        }
        return super().generate(text, **forwarded)


__all__ = ["OsharaXTTSV2Backend"]