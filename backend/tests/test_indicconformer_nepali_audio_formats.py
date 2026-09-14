"""Regression: the IndicConformer sidecar must decode what the app records.

Its engine venv carries a plain libsndfile, which reads WAV/FLAC/OGG but NOT
WebM/Opus — and browser dictation records exactly WebM/Opus. So the sidecar
raised LibsndfileError on the app's own microphone audio, the capture session
surfaced an empty transcript, and the widget rendered that as the generic
"No speech detected" with nothing pointing at the real cause.

The parent already resolves an ffmpeg (bundled static binary, user override,
or system), so the backend publishes it as FFMPEG_PATH and the sidecar falls
back to it for any container libsndfile rejects.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "engines" / "indicconformer_nepali"


def _load_sidecar():
    spec = importlib.util.spec_from_file_location(
        "_indic_sidecar_fmt", ENGINE / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backend_publishes_an_ffmpeg_path_for_the_sidecar(monkeypatch):
    """_spawn must hand the child a decoder before it starts."""
    from engines.indicconformer_nepali import IndicConformerNepaliBackend

    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr(
        "services.ffmpeg_utils.find_ffmpeg", lambda: "/opt/ffmpeg/bin/ffmpeg"
    )
    # Stop at the env work — actually spawning would need the engine venv.
    monkeypatch.setattr(
        "services.subprocess_asr.SubprocessASRBackend._spawn",
        lambda self: None,
    )

    IndicConformerNepaliBackend._spawn(IndicConformerNepaliBackend())

    assert os.environ["FFMPEG_PATH"] == "/opt/ffmpeg/bin/ffmpeg"


def test_an_existing_ffmpeg_path_is_not_overridden(monkeypatch):
    """A user's explicit override (or Tauri's bundled sidecar) wins."""
    from engines.indicconformer_nepali import IndicConformerNepaliBackend

    monkeypatch.setenv("FFMPEG_PATH", "/user/choice/ffmpeg")
    monkeypatch.setattr(
        "services.ffmpeg_utils.find_ffmpeg",
        lambda: pytest.fail("must not re-resolve when FFMPEG_PATH is already set"),
    )
    monkeypatch.setattr(
        "services.subprocess_asr.SubprocessASRBackend._spawn",
        lambda self: None,
    )

    IndicConformerNepaliBackend._spawn(IndicConformerNepaliBackend())

    assert os.environ["FFMPEG_PATH"] == "/user/choice/ffmpeg"


def test_decode_without_any_ffmpeg_explains_itself(monkeypatch):
    """No decoder anywhere → a message naming the fix, not a bare OSError."""
    sidecar = _load_sidecar()
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        sidecar._decode_via_ffmpeg("/tmp/whatever.webm")
