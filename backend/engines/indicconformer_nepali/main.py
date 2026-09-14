"""Length-prefixed ASR sidecar for AI4Bharat IndicConformer Nepali."""
from __future__ import annotations

import contextlib
import json
import os
import struct
import sys
import tempfile
import threading
import traceback
from pathlib import Path

MAX_FRAME_BYTES = 64 * 1024 * 1024
MODEL_ID = "ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large"
MODEL_ARCHIVE = "indicconformer_stt_ne_hybrid_rnnt_large.nemo"

#: Seconds between keep-alive progress frames during a long blocking call.
_HEARTBEAT_S = 5.0

#: Serializes _send across threads (the heartbeat below + the main loop) so
#: concurrent length+body writes can't interleave and corrupt the framing.
_send_lock = threading.Lock()

_model = None


def _prepare_nemo_import() -> None:
    """Bridge NeMo's optional Hub search type removed in newer hub releases."""
    import numpy as np
    import huggingface_hub

    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.bytes_, np.str_],
        }

    if not hasattr(huggingface_hub, "ModelFilter"):
        class ModelFilter:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        huggingface_hub.ModelFilter = ModelFilter

    import pytorch_lightning.loggers as lightning_loggers

    if not hasattr(lightning_loggers, "NeptuneLogger"):
        class NeptuneLogger:
            pass

        lightning_loggers.NeptuneLogger = NeptuneLogger


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with _send_lock:
        stream.write(struct.pack("!I", len(body)))
        stream.write(body)
        stream.flush()


@contextlib.contextmanager
def _heartbeat(stdout, stage: str):
    """Emit a progress frame every ~5s for the duration of the block.

    The first transcription downloads ~500 MB of weights and then restores
    them inside one blocking NeMo call that says nothing on the wire. The
    parent reads that silence two ways, and both kill a healthy cold start:
    ``SubprocessASRBackend.transcribe`` re-arms its recv watchdog on every
    frame, and each frame also reports activity to the GPU pool's execution
    clock. Percent climbs 1..99 because the upstream call exposes no real
    progress; it is a liveness signal, not a measurement.
    """
    stop = threading.Event()

    def _beat() -> None:
        pct = 1
        while not stop.wait(_HEARTBEAT_S):
            pct = min(pct + 1, 99)
            try:
                _send(stdout, {"op": "progress", "stage": stage, "percent": pct})
            except Exception:
                return  # pipe gone — the main loop will surface it

    hb = threading.Thread(
        target=_beat, name=f"indicconformer-{stage}-heartbeat", daemon=True
    )
    hb.start()
    try:
        yield
    finally:
        stop.set()
        hb.join(timeout=_HEARTBEAT_S + 1)


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (size,) = struct.unpack("!I", header)
    if size > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {size}")
    body = stream.read(size)
    if len(body) != size:
        raise IOError("short read")
    return json.loads(body.decode("utf-8"))


def _load_model(stdout):
    global _model
    if _model is not None:
        return _model

    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 0})
    with _heartbeat(stdout, "loading_model"):
        import torch
        _prepare_nemo_import()
        import nemo.collections.asr as nemo_asr

        # This gated repository publishes a single .nemo archive rather than
        # the model_config.yaml layout expected by NeMo's from_pretrained
        # helper.
        archive = next(
            Path.home().glob(f".cache/torch/NeMo/**/{MODEL_ARCHIVE}"),
            None,
        )
        if archive is None:
            from huggingface_hub import hf_hub_download
            archive = Path(hf_hub_download(MODEL_ID, MODEL_ARCHIVE))
        if not archive.is_file():
            raise FileNotFoundError(
                f"IndicConformer archive not found: {archive.name}"
            )
        _model = nemo_asr.models.ASRModel.restore_from(
            str(archive), map_location="cpu"
        )
        _model.freeze()
        _model = _model.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _model


def _decode_via_ffmpeg(audio_path: str) -> tuple[str, str]:
    """Decode any container to 16 kHz mono PCM_16 WAV using the parent's ffmpeg.

    libsndfile (what soundfile wraps) covers WAV/FLAC/OGG but not WebM/Opus,
    MP4/AAC or MP3-in-MP4 — and dictation's browser capture is WebM/Opus, so
    without this the engine rejects the app's own microphone audio. The parent
    publishes its resolved binary as FFMPEG_PATH when it spawns us.
    """
    import shutil
    import subprocess

    ffmpeg = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "this audio format needs ffmpeg to decode, and none was found — "
            "install ffmpeg or set FFMPEG_PATH"
        )
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", audio_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", handle.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=120, check=True,
        )
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return handle.name, handle.name


def _prepare_audio(audio_path: str) -> tuple[str, str | None, float]:
    """Convert input audio to the 16 kHz mono WAV required by the model.

    Also returns the clip duration in seconds. The parent's segmenter derives
    word timings from the segment's end time, and treats a missing end as
    0.1 s (`segmentation._words_from_whisper`) — so reporting the real
    duration is what keeps a transcript spread across the whole clip instead
    of collapsing into its first tenth of a second.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    decoded_path: str | None = None
    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception:
        # Not a container libsndfile understands (WebM/Opus, MP4/AAC, ...).
        audio_path, decoded_path = _decode_via_ffmpeg(audio_path)
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = resample_poly(audio, 16000, sample_rate)
    duration = len(audio) / 16000.0
    # Already the mono 16 kHz PCM the model wants — transcribe it in place
    # rather than paying a full decode+rewrite per request.
    if sample_rate == 16000 and audio_path.lower().endswith(".wav"):
        info = sf.info(audio_path)
        if info.channels == 1 and info.subtype == "PCM_16":
            return audio_path, decoded_path, duration
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    sf.write(handle.name, audio, 16000, subtype="PCM_16")
    if decoded_path:
        # The ffmpeg intermediate has been resampled into `handle`; drop it now
        # rather than leaking one temp WAV per non-libsndfile transcription.
        try:
            os.unlink(decoded_path)
        except OSError:
            pass
    return handle.name, handle.name, duration


def _normalize_result(outputs, duration: float) -> dict:
    """Shape a NeMo hypothesis into the Whisper-like dict the parent expects.

    This model emits one utterance with no word or segment timings, so the
    single segment spans the whole clip (the Moonshine backend does the same).
    ``text`` is carried at the top level too: it is the parent segmenter's
    fallback when a result has no usable word timings.
    """
    hypothesis = outputs[0] if outputs else None
    if hypothesis is None:
        return {"segments": [], "chunks": [], "text": "", "language": "ne"}
    raw_text = getattr(hypothesis, "text", None) or hypothesis
    if isinstance(raw_text, (list, tuple)):
        text = " ".join(str(item) for item in raw_text).strip()
    else:
        text = str(raw_text).strip()
    end = round(float(duration), 3) if duration > 0 else None
    segment = {"text": text, "start": 0.0, "end": end, "words": []}
    return {
        "segments": [segment] if text else [],
        "chunks": [{"text": text, "timestamp": (0.0, end)}] if text else [],
        "text": text,
        "language": "ne",
    }


def _handle_transcribe(msg: dict, stdout) -> None:
    audio_path = msg.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("transcribe: missing 'audio_path'")

    model = _load_model(stdout)
    prepared_path, cleanup_path, duration = _prepare_audio(audio_path)
    try:
        # CTC is the stable decoder recommended by the model card for Nepali.
        model.cur_decoder = str(msg.get("decoder") or "ctc").lower()
        with _heartbeat(stdout, "transcribing"):
            outputs = model.transcribe(
                [prepared_path],
                batch_size=1,
                logprobs=False,
                language_id="ne",
            )
        _send(stdout, {"op": "segments", "result": _normalize_result(outputs, duration)})
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass


def main() -> int:
    stdin = sys.stdin.buffer
    frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(frame_fd, "wb")
    _send(stdout, {"op": "ready", "engine": "indicconformer-nepali"})
    while True:
        try:
            msg = _recv(stdin)
        except Exception as exc:
            _send(stdout, {"op": "error", "stage": "recv", "message": str(exc)})
            return 1
        if msg is None:
            return 0
        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "ping":
                _send(stdout, {"op": "pong", "vram_mb": 0})
            elif op == "transcribe":
                _handle_transcribe(msg, stdout)
            elif op == "shutdown":
                return 0
            else:
                _send(stdout, {"op": "error", "stage": "dispatch", "message": f"unknown op: {op!r}"})
        except Exception as exc:
            _send(stdout, {
                "op": "error",
                "stage": op or "unknown",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    sys.exit(main())