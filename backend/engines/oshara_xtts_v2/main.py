"""Length-prefixed stdio sidecar for Oshara Nepali XTTS-v2.

Sampling defaults reproduce a plain ``inference(text, language=...,
temperature=0.7)`` call. ``OMNIVOICE_OSHARA_XTTS_ROUTE`` picks how Nepali
reaches the GPT: ``hi`` (default) sends it as plain Hindi, ``ne`` runs the
Hindi cleaners while keeping the ``ne`` code, as the model card implies.
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import sys
import threading
import traceback
from collections import OrderedDict
from pathlib import Path

MAX_FRAME_BYTES = 64 * 1024 * 1024
SAMPLE_RATE = 24000

#: How Nepali reaches the GPT — "hi" (plain Hindi) or "ne" (Hindi cleaners,
#: "ne" code). The tokenizer ships no "ne" entry, so "ne" needs the patch below.
_NE_ROUTE = os.environ.get("OMNIVOICE_OSHARA_XTTS_ROUTE", "hi").strip().lower()

#: Emit a progress frame at least this often so the parent's recv watchdog and
#: the GPU pool's execution clock both read a slow CPU synth as alive rather
#: than wedged. Same reason the ASR sidecar heartbeats.
_HEARTBEAT_S = 5.0

#: Silence inserted between separately-synthesised sentences, in seconds.
_GAP_S = 0.2

#: ref_audio must be a local file path, never a URL: this sidecar runs with the
#: user's filesystem access, and fetching a caller-supplied URL would make it an
#: SSRF vector and break the local-first guarantee.
_URL_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)

#: Conditioning latents are expensive to compute; cache a few, keyed by file
#: identity (path + mtime + size) so re-recording a clip re-conditions.
_VOICE_CACHE_MAX = 8

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
#: Split after danda / double danda / terminal punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u0964\u0965?!])\s*|(?<=\.)\s+")

_model = None
_voice_cache: "OrderedDict[str, tuple]" = OrderedDict()

#: Serialises _send across the heartbeat thread and the main loop so two
#: concurrent length+body writes cannot interleave and corrupt the framing.
_send_lock = threading.Lock()


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with _send_lock:
        stream.write(struct.pack("!I", len(body)))
        stream.write(body)
        stream.flush()


class _Heartbeat:
    """Emit a progress frame every ~5s for the duration of the block."""

    def __init__(self, stdout, stage: str) -> None:
        self._stdout = stdout
        self._stage = stage
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"oshara-{stage}-heartbeat", daemon=True
        )
        self.percent = 0

    def _frame(self) -> dict:
        return {"op": "progress", "stage": self._stage, "percent": self.percent}

    def _run(self) -> None:
        while not self._stop.wait(_HEARTBEAT_S):
            try:
                _send(self._stdout, self._frame())
            except Exception:
                return  # pipe gone — the main loop surfaces it

    def __enter__(self) -> "_Heartbeat":
        _send(self._stdout, self._frame())
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=_HEARTBEAT_S + 1)


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


def _load_reference_audio(path: str, sampling_rate: int):
    """Load reference audio without torchaudio's TorchCodec dependency."""
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_rate != sampling_rate:
        audio = resample_poly(audio, sampling_rate, source_rate)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    return torch.from_numpy(audio.copy()).unsqueeze(0)


def _load_model(stdout):
    global _model
    if _model is not None:
        return _model

    # The first call downloads ~1.9 GB and restores the checkpoint inside one
    # blocking call that says nothing on the wire; heartbeat so the parent can
    # tell a healthy cold load from a wedged sidecar.
    with _Heartbeat(stdout, "loading_model"):
        import torch
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        import TTS.tts.models.xtts as xtts_module
        from TTS.tts.models.xtts import Xtts

        # Coqui's current loader routes through torchcodec, whose wheel in the
        # sibling environment expects an unavailable CUDA runtime library even
        # on CPU. XTTS only needs a mono 22.05 kHz tensor here, so use
        # soundfile.
        xtts_module.load_audio = _load_reference_audio

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        root = snapshot_download(
            "Oshara/xtts-v2-nepali",
            allow_patterns=["epoch-10/*"],
        )
        model_dir = os.path.join(root, "epoch-10")
        config = XttsConfig()
        config.load_json(os.path.join(model_dir, "config.json"))
        _model = Xtts.init_from_config(config)
        _model.load_checkpoint(config, checkpoint_dir=model_dir, use_deepspeed=False)
        _model.to(device)
        _patch_nepali_tokenizer(_model.tokenizer)
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _model


def _patch_nepali_tokenizer(tokenizer) -> None:
    """Let ``ne`` through coqui-tts's tokenizer using the Hindi cleaners.

    The shipped tokenizer knows 17 languages and ``ne`` is not one of them, so
    routing Nepali as ``ne`` would raise. Reuse the Hindi cleaners (same script)
    and borrow Hindi's character limit.
    """
    original = tokenizer.preprocess_text

    def preprocess_text(txt, lang):
        return original(txt, "hi" if lang == "ne" else lang)

    tokenizer.preprocess_text = preprocess_text
    tokenizer.char_limits.setdefault("ne", tokenizer.char_limits.get("hi", 150))


def _reference_path() -> str:
    """The bundled fallback clip, as an absolute path when it can be found.

    A bare relative name would resolve against the sidecar's working directory
    rather than the project, so return None instead of a path that cannot be
    opened — the caller then reports "no reference clip" rather than ENOENT on
    a mystery filename.
    """
    default_root = Path(__file__).resolve().parents[4] / "oshara_xtts_v2"
    root = Path(
        os.environ.get("OMNIVOICE_OSHARA_XTTS_DIR", str(default_root))
    ).expanduser()
    candidate = root / "test_voice.wav"
    return str(candidate) if candidate.is_file() else ""


def _voice_latents(model, ref_audio: str):
    """(gpt_cond_latent, speaker_embedding) for a local reference clip.

    Cached per file identity (path + mtime + size), so re-recording a clip at
    the same path re-conditions rather than serving stale latents. LRU-bounded
    because each entry holds GPT conditioning tensors.
    """
    if not ref_audio:
        raise ValueError(
            "No reference clip given and no bundled test_voice.wav found; "
            "pick a voice to clone."
        )
    if _URL_RE.match(ref_audio):
        raise ValueError(
            "ref_audio must be a local file path; URLs are not accepted "
            "(local-first)."
        )
    st = os.stat(ref_audio)
    key = f"{ref_audio}|m{st.st_mtime_ns}s{st.st_size}"
    cached = _voice_cache.get(key)
    if cached is not None:
        _voice_cache.move_to_end(key)
        return cached
    latents = model.get_conditioning_latents(audio_path=[ref_audio])
    _voice_cache[key] = latents
    if len(_voice_cache) > _VOICE_CACHE_MAX:
        _voice_cache.popitem(last=False)
    return latents


def _chunks(text: str, limit: int) -> list[str]:
    """Sentence pieces no longer than ``limit`` characters, speakable only.

    Past the tokenizer's per-language character limit the GPT drifts or
    truncates, so long text is split on the danda / double danda / terminal
    punctuation and synthesised piece by piece.
    """
    text = re.sub(r"\s+", " ", _ZERO_WIDTH_RE.sub("", text)).strip()
    if len(text) <= limit:
        return [text] if any(ch.isalnum() for ch in text) else []
    out: list[str] = []
    for sent in _SENTENCE_SPLIT_RE.split(text):
        sent = (sent or "").strip()
        while len(sent) > limit:
            cut = max(sent.rfind(",", 0, limit), sent.rfind(" ", 0, limit))
            if cut <= 0:
                cut = limit - 1
            head, sent = sent[: cut + 1].strip(), sent[cut + 1 :].strip()
            if head:
                out.append(head)
        if sent:
            out.append(sent)
    return [c for c in out if any(ch.isalnum() for ch in c)]


def _handle_synthesize(msg: dict, stdout) -> None:
    text = msg.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("synthesize: missing or empty 'text'")

    model = _load_model(stdout)

    import numpy as np
    import torch

    # XTTS-v2 has no Nepali token. "hi" (default) sends Nepali as plain Hindi;
    # "ne" keeps the Nepali code and relies on the tokenizer patch above. Same
    # script either way, so both are intelligible — this picks which the model
    # card's guidance you follow.
    route = "ne" if _NE_ROUTE == "ne" else "hi"
    limit = int(getattr(model.tokenizer, "char_limits", {}).get(route, 150))
    pieces_text = _chunks(text, limit)
    if not pieces_text:
        raise ValueError("synthesize: text has nothing speakable")

    ref_audio = msg.get("ref_audio") or _reference_path()
    gpt_cond_latent, speaker_embedding = _voice_latents(model, ref_audio)

    temperature = float(msg.get("temperature", 0.7))
    length_penalty = float(msg.get("length_penalty", 1.0))
    gap = np.zeros(int(_GAP_S * SAMPLE_RATE), dtype=np.float32)
    pieces: list = []
    with _Heartbeat(stdout, "synthesizing") as hb, torch.inference_mode():
        for i, piece in enumerate(pieces_text):
            hb.percent = int(100 * i / len(pieces_text))
            output = model.inference(
                text=piece,
                language=route,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                length_penalty=length_penalty,
                enable_text_splitting=False,
            )
            if pieces:
                pieces.append(gap)
            pieces.append(
                np.asarray(output["wav"], dtype=np.float32).reshape(-1)
            )

    audio = np.clip(np.concatenate(pieces), -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    _send(stdout, {
        "op": "audio",
        "audio_pcm_b64": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": SAMPLE_RATE,
        "n_samples": int(audio.shape[0]),
    })


def main() -> int:
    stdin = sys.stdin.buffer
    frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(frame_fd, "wb")
    _send(stdout, {"op": "ready", "engine": "oshara-xtts-v2", "sample_rate": SAMPLE_RATE})
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
            elif op == "synthesize":
                _handle_synthesize(msg, stdout)
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