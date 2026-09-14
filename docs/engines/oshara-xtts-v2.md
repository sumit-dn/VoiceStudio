# Oshara XTTS-v2

VoiceStudio can use the sibling `oshara_xtts_v2` project as an isolated
subprocess engine for Nepali speech synthesis and reference-voice cloning.

> [!WARNING]
> **Non-commercial weights.** `Oshara/xtts-v2-nepali` inherits the
> [Coqui Public Model License](https://coqui.ai/cpml). This is a local try-out
> engine: it does not meet the licence bar in
> [engine-acceptance.md](../engine-acceptance.md) (#2, "Licence clean for
> commercial use — model weights *and* code"), so it stays opt-in and is not an
> upstream candidate as-is.

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

## Settings

| Env var | Default | Effect |
|---|---|---|
| `OMNIVOICE_OSHARA_XTTS_DIR` | `../oshara_xtts_v2` | The sibling project holding `.venv` and `test_voice.wav` |
| `OMNIVOICE_OSHARA_XTTS_ROUTE` | `hi` | How Nepali reaches the GPT. `hi` sends it as plain Hindi; `ne` keeps the Nepali code and runs the Hindi cleaners |

XTTS-v2 has no Nepali token, so Nepali must ride one of the base languages.
Both routes use the same Devanagari cleaners and are intelligible; `hi` is the
default because it needs no tokenizer patching.

Text longer than the tokenizer's per-language character limit (150 for Hindi)
is split on the danda `।`, double danda `॥` and terminal punctuation, then
synthesised sentence by sentence with a short gap between pieces — past that
limit the model drifts or truncates.

## Voice input

When no reference clip is supplied, the engine uses `test_voice.wav` from the
Oshara project. A supplied reference clip is conditioned and reused by the
sidecar until a different clip is selected.