"""Regression: the Oshara XTTS sidecar's pure helpers.

These cover the three things the first cut of this sidecar got wrong, each of
which only shows up at synthesis time and none of which needs the 1.9 GB
checkpoint to test:

* Long text went to the GPT in one pass. Past the tokenizer's per-language
  character limit XTTS drifts or truncates, so anything over the limit must be
  split on Devanagari sentence punctuation.
* ``ref_audio`` was passed through unchecked. The sidecar runs with the user's
  filesystem access, so a caller-supplied URL would make it an SSRF vector and
  break the local-first guarantee.
* The no-reference fallback returned a bare relative ``test_voice.wav``, which
  resolves against the sidecar's working directory rather than the project —
  an ENOENT on a mystery filename instead of a usable message.

Loaded by path: the sidecar is written for the engine's own venv (coqui-tts +
transformers 4.57) and is never importable as a package from the app venv.
Only stdlib-only helpers are exercised here.
"""
import importlib.util
from pathlib import Path

import pytest

SIDECAR = (
    Path(__file__).resolve().parents[1]
    / "engines" / "oshara_xtts_v2" / "main.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_oshara_sidecar", SIDECAR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestChunking:
    def test_text_within_the_limit_is_one_piece(self):
        m = _load()
        assert m._chunks("नमस्ते संसार।", 150) == ["नमस्ते संसार।"]

    def test_long_text_splits_on_the_danda_and_respects_the_limit(self):
        m = _load()
        limit = 150
        pieces = m._chunks("यो पहिलो वाक्य हो। " * 20, limit)

        assert len(pieces) > 1, "text past the limit must be split"
        assert all(len(p) <= limit for p in pieces)

    def test_a_single_sentence_longer_than_the_limit_still_splits(self):
        """No danda to split on — fall back to a comma/space break."""
        m = _load()
        limit = 40
        pieces = m._chunks("शब्द " * 60, limit)

        assert pieces
        assert all(len(p) <= limit for p in pieces)

    def test_unspeakable_text_yields_nothing(self):
        m = _load()
        assert m._chunks("   ...   ", 150) == []
        assert m._chunks("", 150) == []


class TestReferenceAudioGuard:
    @pytest.mark.parametrize(
        "url",
        ["http://evil.test/x.wav", "https://evil.test/x.wav", "file:///etc/passwd"],
    )
    def test_a_url_reference_is_refused(self, url):
        m = _load()
        with pytest.raises(ValueError, match="local file path"):
            m._voice_latents(object(), url)

    def test_no_reference_says_so_instead_of_guessing_a_filename(self):
        m = _load()
        with pytest.raises(ValueError, match="pick a voice to clone"):
            m._voice_latents(object(), "")

    def test_reference_path_returns_empty_when_the_clip_is_absent(self, monkeypatch, tmp_path):
        """Never hand back a bare relative name the sidecar cannot open."""
        m = _load()
        monkeypatch.setenv("OMNIVOICE_OSHARA_XTTS_DIR", str(tmp_path))

        assert m._reference_path() == ""

    def test_reference_path_finds_the_bundled_clip(self, monkeypatch, tmp_path):
        m = _load()
        (tmp_path / "test_voice.wav").write_bytes(b"RIFF")
        monkeypatch.setenv("OMNIVOICE_OSHARA_XTTS_DIR", str(tmp_path))

        assert m._reference_path() == str(tmp_path / "test_voice.wav")


def test_voice_cache_is_lru_bounded_and_keyed_by_file_identity(tmp_path):
    """Re-recording a clip at the same path must re-condition, not serve stale."""
    m = _load()
    calls = []

    class _Model:
        def get_conditioning_latents(self, audio_path):
            calls.append(audio_path[0])
            return ("latent", len(calls))

    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"a" * 100)
    model = _Model()

    first = m._voice_latents(model, str(clip))
    assert m._voice_latents(model, str(clip)) == first
    assert len(calls) == 1, "second call must hit the cache"

    # Same path, different contents -> different mtime/size -> recondition.
    clip.write_bytes(b"b" * 200)
    second = m._voice_latents(model, str(clip))
    assert second != first
    assert len(calls) == 2

    # Bounded.
    for i in range(m._VOICE_CACHE_MAX + 4):
        other = tmp_path / f"c{i}.wav"
        other.write_bytes(b"x" * (i + 1))
        m._voice_latents(model, str(other))
    assert len(m._voice_cache) <= m._VOICE_CACHE_MAX
