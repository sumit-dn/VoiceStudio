"""Length-prefixed stdio sidecar for Oshara Nepali XTTS-v2."""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import traceback
from pathlib import Path

MAX_FRAME_BYTES = 64 * 1024 * 1024
SAMPLE_RATE = 24000
_model = None
_conditioning = None


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!I", len(body)))
    stream.write(body)
    stream.flush()


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

    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 0})
    import torch
    from huggingface_hub import snapshot_download
    from TTS.tts.configs.xtts_config import XttsConfig
    import TTS.tts.models.xtts as xtts_module
    from TTS.tts.models.xtts import Xtts

    # Coqui's current loader routes through torchcodec, whose wheel in the
    # sibling environment expects an unavailable CUDA runtime library even on
    # CPU. XTTS only needs a mono 22.05 kHz tensor here, so use soundfile.
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
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _model


def _reference_path() -> str:
    default_root = Path(__file__).resolve().parents[4] / "oshara_xtts_v2"
    root = Path(
        os.environ.get("OMNIVOICE_OSHARA_XTTS_DIR", str(default_root))
    ).expanduser()
    if (root / "test_voice.wav").is_file():
        return str(root / "test_voice.wav")
    return "test_voice.wav"


def _load_conditioning(model, ref_audio: str):
    global _conditioning
    if _conditioning is None or _conditioning[0] != ref_audio:
        latents = model.get_conditioning_latents(audio_path=[ref_audio])
        _conditioning = (ref_audio, *latents)
    return _conditioning[1:]


def _handle_synthesize(msg: dict, stdout) -> None:
    text = msg.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("synthesize: missing or empty 'text'")

    model = _load_model(stdout)
    ref_audio = msg.get("ref_audio") or _reference_path()
    gpt_cond_latent, speaker_embedding = _load_conditioning(model, ref_audio)
    output = model.inference(
        text=text,
        language="hi",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=float(msg.get("temperature", 0.7)),
        length_penalty=float(msg.get("length_penalty", 1.0)),
    )
    import numpy as np

    audio = np.asarray(output["wav"], dtype=np.float32).squeeze()
    audio = np.clip(audio, -1.0, 1.0)
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