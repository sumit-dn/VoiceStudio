# Oshara XTTS-v2

VoiceStudio can use the sibling `oshara_xtts_v2` project as an isolated
subprocess engine for Nepali speech synthesis and reference-voice cloning.

## Setup

The default location is:

```text
../oshara_xtts_v2
```

From that project, install its existing environment:

```bash
cd ../oshara_xtts_v2
uv sync
```

If the project is elsewhere, set its path before starting VoiceStudio:

```bash
export OMNIVOICE_OSHARA_XTTS_DIR=/absolute/path/to/oshara_xtts_v2
```

The `Oshara XTTS-v2 (Nepali)` engine will then appear in Model Catalogue.
The first synthesis downloads the `Oshara/xtts-v2-nepali` `epoch-10` checkpoint
from Hugging Face. CPU inference works but is considerably slower than CUDA.

## Voice input

When no reference clip is supplied, the engine uses `test_voice.wav` from the
Oshara project. A supplied reference clip is conditioned and reused by the
sidecar until a different clip is selected.