# IndicConformer Nepali ASR

VoiceStudio supports the AI4Bharat model:

```text
ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large
```

It accepts 16 kHz mono audio and uses the CTC decoder by default with Nepali
language ID `ne`. The model is gated on Hugging Face, so accept its access
conditions and authenticate the dedicated environment before first use.

## Install in an isolated environment

Do not install AI4Bharat NeMo into VoiceStudio's shared `.venv`; its dependency
pins can conflict with VoiceStudio. Create a dedicated environment instead:

```bash
export OMNIVOICE_INDICCONFORMER_ASR_DIR="$HOME/.omnivoice/engines/indicconformer-nepali"
mkdir -p "$OMNIVOICE_INDICCONFORMER_ASR_DIR"
cd "$OMNIVOICE_INDICCONFORMER_ASR_DIR"
uv venv --python 3.11 .venv
source .venv/bin/activate
git clone https://github.com/AI4Bharat/NeMo.git nemo-source
cd nemo-source
git checkout nemo-v2
bash reinstall.sh
```

The `reinstall.sh` script installs the AI4Bharat NeMo dependencies into the
active environment. Restart VoiceStudio after installation, then select
**IndicConformer (AI4Bharat Nepali)** in the ASR engine catalogue or set:

```bash
export OMNIVOICE_ASR_BACKEND=indicconformer-nepali
```

The model downloads from Hugging Face on first transcription. The engine
normalizes uploaded audio to the required 16 kHz mono WAV automatically.

## Hugging Face authorization

Open the model page, sign in, and accept its access conditions:

<https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large>

Then authenticate from the dedicated environment without placing the token in
VoiceStudio's shared environment:

```bash
export OMNIVOICE_INDICCONFORMER_ASR_DIR="$HOME/.omnivoice/engines/indicconformer-nepali"
"$OMNIVOICE_INDICCONFORMER_ASR_DIR/.venv/bin/hf" auth login
```

Alternatively, set `HF_TOKEN` only in the shell that starts VoiceStudio. The
sidecar inherits that environment when it downloads the model.

## Audio formats

The engine's environment ships a plain libsndfile, which reads WAV, FLAC and
OGG but not WebM/Opus or MP4/AAC. VoiceStudio publishes its resolved ffmpeg to
the sidecar as `FFMPEG_PATH`, so any container ffmpeg can decode works —
including the WebM/Opus that browser dictation records. An explicit
`FFMPEG_PATH` in the environment that starts VoiceStudio is never overridden.

If no ffmpeg resolves at all, WAV/FLAC/OGG still transcribe and other formats
fail with a message naming the missing dependency.

## Transcribing a file

Drag an audio file onto the Transcriptions page, or use **Browse files**.
Files are transcribed one at a time and land in the same history as dictation.
The equivalent API call:

```bash
curl -F audio=@clip.m4a -F mode=accurate http://127.0.0.1:3900/transcribe
```

## Dictation

The engine works with the microphone dictation hotkey as well as file drops.
Select **IndicConformer (AI4Bharat Nepali)** in Settings → Dictation.

Loading the model takes roughly 30 seconds, so VoiceStudio's background
capture-ASR preload warms it shortly after startup; dictation then finalises in
well under a second. Starting a dictation before the preload has finished is
safe — the first session simply waits for that one load.

Set `OMNIVOICE_CAPTURE_PRELOAD_DELAY` to change how long after startup the
preload runs. The preload is skipped when free RAM is under 4 GB, in which
case the model loads on first use instead.
