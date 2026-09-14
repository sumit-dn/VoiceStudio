"""Crash-isolated ASR backends (Wave 4.2 / Spec 7).

Native ASR engines (the whisper.cpp / CTranslate2 class) can segfault on GPU
teardown — a process-level crash that takes the whole backend down with it.
Running them in a child process turns that segfault into a *failed job*: the
sidecar dies, the parent surfaces a decorated error, and the next request
respawns a fresh sidecar.

This reuses ``SubprocessBackend``'s wire protocol + lifecycle (spawn, ready
handshake, length-prefixed JSON, GPU-slot acquire, and — critically —
respawn-on-dead-process: ``_spawn`` relaunches whenever the previous child
isn't alive). We add a ``transcribe`` op alongside the TTS ``synthesize`` op;
the TTS ``generate`` surface is stubbed since an ASR sidecar never synthesizes.

The base is engine-agnostic; concrete subclasses point ``sidecar_script()`` at
an engine runner. ``IsolatedFasterWhisperBackend`` wraps faster-whisper (the
CTranslate2 engine with the documented GPU-teardown crash) using the parent
venv — faster-whisper is already a dependency, so no separate venv is needed,
only process isolation.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from services.subprocess_backend import (
    RECV_TIMEOUT_S,
    SubprocessBackend,
)

logger = logging.getLogger("omnivoice.asr.subprocess")

# A model load + transcription can take a while on CPU for a long clip; give
# the transcribe round-trip more headroom than the TTS default.
ASR_RECV_TIMEOUT_S = 600.0


class SubprocessASRBackend(SubprocessBackend):
    """Crash-isolated ASR over the SubprocessBackend protocol.

    Concrete subclasses set ``id`` / ``display_name`` and override
    ``venv_python()`` / ``sidecar_script()``. They are registered in the ASR
    registry (``services.asr_backend._REGISTRY``); the registry uses
    ``is_available()`` + ``transcribe()`` duck-typed, so subclassing the TTS
    ``SubprocessBackend`` is fine.
    """

    # ── TTS surface stubs (an ASR sidecar never synthesizes) ───────────────
    @property
    def sample_rate(self) -> int:  # pragma: no cover - unused
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:  # pragma: no cover - unused
        return ["multi"]

    def generate(self, text: str, **kw):  # pragma: no cover - unused
        raise NotImplementedError("ASR sidecar does not synthesize speech")

    def warmup(self) -> None:
        """Load the model NOW, off the user's first dictation (#888 class).

        The background capture-ASR preload calls this — but only when the
        backend actually has it. Neither subprocess ASR engine did, so the
        preload selected one, found no ``warmup``, and silently did nothing;
        the whole cold load then landed inside the user's first dictation
        session, which sat there for tens of seconds looking hung (measured
        here: 29 s first call, 0.4 s after).

        Spawning alone is not enough: these sidecars import torch and restore
        weights lazily, on their first transcribe. So warm with a real (tiny,
        silent) transcription — that is the only thing that proves the model
        is resident. Best-effort by contract: the preload runs in the
        background and a failure here must never break dictation, which
        simply falls back to loading on first use.
        """
        import tempfile
        import time
        import wave

        t0 = time.perf_counter()
        tmp = None
        try:
            with self._lock:
                self._spawn()
            # 0.1 s of silence at 16 kHz mono — every ASR engine accepts it,
            # and it costs nothing next to the model load it triggers.
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            tmp = handle.name
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 1600)
            self.transcribe(tmp, word_timestamps=False)
            logger.info(
                "%s ASR sidecar warmed up in %.1fs",
                self.id, time.perf_counter() - t0,
            )
        except Exception as e:  # noqa: BLE001 — preload must never break dictation
            logger.warning(
                "%s ASR warmup failed after %.1fs (will load on first use): %s",
                self.id, time.perf_counter() - t0, e,
            )
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def ensure_loaded(self) -> None:
        """ASRBackend's eager-load hook (the ASR contract's name for it).

        These backends inherit the TTS ``SubprocessBackend``, which spells the
        same operation ``ensure_ready`` — so the ASR loader
        (``load_active_asr_backend``, used by the batch and "accurate"
        transcribe paths) hit AttributeError on EVERY subprocess ASR engine
        and took the request down with it. The ASR surface is duck-typed
        against ``ASRBackend``, so the contract has to be satisfied by name.

        Spawning the sidecar here is what the hook is for: it moves the cold
        model load out of the caller's transcribe budget and into the
        model-load budget, and surfaces a broken engine as one clean error up
        front rather than a cryptic failure mid-transcription.
        """
        with self._lock:
            self._spawn()

    # ── ASR surface ────────────────────────────────────────────────────────
    @staticmethod
    def _device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def transcribe(self, audio_path: str, *, word_timestamps: bool = True) -> dict:
        """Transcribe ``audio_path`` in the sidecar. Returns the engine's
        result dict ({"segments": [...], "language": ...}).

        A sidecar crash mid-transcription raises a RuntimeError decorated with
        the engine id + device (so the failure is attributable, not a bare
        broken-pipe) — and the *next* call respawns a fresh sidecar via
        ``_spawn``'s dead-process check. Acquires a GPU-pool slot for the
        duration, released even if the child dies (the base's try/finally)."""
        # On-pool callers (run_transcribe_guarded dispatches via run_in_executor
        # on the GPU pool) already own a pool slot; re-acquiring would
        # self-deadlock on a 1-worker (MPS) pool, so skip it. Off-pool callers
        # hold a real slot for the whole transcription via _occupy. Mirrors
        # SubprocessBackend.generate()'s path-aware slot block.
        from services.model_manager import running_on_gpu_pool
        _held = None
        slot_future = None
        if not running_on_gpu_pool():
            from services.model_manager import _get_gpu_pool
            pool = _get_gpu_pool()
            _held = threading.Event()
            _acquired = threading.Event()

            def _occupy():
                _acquired.set()
                _held.wait()

            slot_future = pool.submit(_occupy)

        try:
            if _held is not None and not _acquired.wait(timeout=10):
                if slot_future is not None:
                    slot_future.cancel()
                raise TimeoutError("timed out waiting for a free GPU worker")
            with self._lock:
                self._spawn()
                self._send({
                    "op": "transcribe",
                    "audio_path": str(audio_path),
                    "word_timestamps": bool(word_timestamps),
                })
                reply = self._recv_with_timeout(ASR_RECV_TIMEOUT_S)
                # A cold sidecar may emit non-terminal {"op": "progress"}
                # frames (a model load, audio preprocessing) before the
                # terminal segments frame. Each recv re-arms the watchdog, so
                # a long-but-active load survives while a silent wedge is
                # still killed at the deadline. Each frame is also reported to
                # the GPU pool's execution clock, exactly as the TTS generate
                # path does (#1367) — without it the outer transcribe budget
                # can expire during a healthy multi-GB cold load and blame the
                # hardware for it.
                while reply is not None and reply.get("op") == "progress":
                    try:
                        from services.model_manager import (
                            report_model_load_activity, running_on_gpu_pool,
                        )
                        if running_on_gpu_pool():
                            report_model_load_activity()
                    except Exception:
                        pass  # heartbeat is best-effort; never fail a job over it
                    reply = self._recv_with_timeout(ASR_RECV_TIMEOUT_S)
            if not reply:
                # Pipe closed mid-transcription → the child crashed.
                raise RuntimeError(
                    f"{self.id} ASR sidecar crashed mid-transcription "
                    f"(device={self._device()}); the job failed but the backend "
                    f"stayed up — retry to respawn a fresh sidecar."
                )
            if reply.get("op") == "error":
                raise RuntimeError(
                    f"{self.id} ASR sidecar error (device={self._device()}): "
                    f"{reply.get('message')!r}"
                )
            if reply.get("op") != "segments":
                raise RuntimeError(
                    f"{self.id} ASR sidecar returned unexpected op: {reply.get('op')!r}"
                )
            return reply.get("result") or {"segments": [], "language": "unknown"}
        finally:
            if _held is not None:
                _held.set()


class IsolatedFasterWhisperBackend(SubprocessASRBackend):
    """faster-whisper (CTranslate2) in a child process — opt-in.

    CTranslate2's GPU teardown can segfault (the endemic faster-whisper crash);
    running it isolated keeps that from killing the backend. Uses the PARENT
    venv (faster-whisper is already installed) — only the process boundary is
    new. Select with ``OMNIVOICE_ASR_BACKEND=faster-whisper-isolated``.
    """

    id = "faster-whisper-isolated"
    display_name = "Faster-Whisper (crash-isolated subprocess)"
    # Same engine as FasterWhisperBackend, so the same device support — the
    # sidecar picks cuda/cpu itself via `_device()`. Without this the registry
    # default ("cpu",) would dishonestly report cpu_only routing on CUDA hosts.
    gpu_compat = ("cuda", "cpu")

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except Exception as e:
            return False, f"faster-whisper not installed: {e}"
        if not cls.sidecar_script().is_file():
            return False, f"ASR sidecar script missing at {cls.sidecar_script()}"
        # Same CTranslate2 engine, same cuDNN 8 requirement (#1371). Crash
        # isolation means a missing cuDNN 8 kills only the child — so instead of
        # a dead backend the user gets a sidecar that fails every transcribe
        # with no explanation. Report it here, where Model Catalogue shows it.
        from services.asr_backend import _ctranslate2_cudnn_ok

        return _ctranslate2_cudnn_ok()

    @classmethod
    def venv_python(cls) -> Path:
        # faster-whisper lives in the parent venv — isolation is process-only.
        return Path(sys.executable)

    @classmethod
    def sidecar_script(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "engines" / "_asr_sidecar" / "main.py"
