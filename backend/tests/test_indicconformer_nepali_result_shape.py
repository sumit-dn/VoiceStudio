"""Regression: the IndicConformer Nepali sidecar must report the real clip
duration, or the whole transcript collapses into the first 0.1 s of the audio.

The model returns one utterance with no timings. The sidecar used to report
``end: None`` for that segment (and ``(0.0, None)`` for the chunk), but the
parent segmenter reads a missing chunk end as ``start + 0.1``
(``segmentation._words_from_whisper``), then spreads every word evenly inside
that window. An 11-second clip therefore came back as a single segment ending
at 0.1 s — subtitles flashed by instantly and dubbing had no room to speak.

Also pins the top-level ``text`` key: it is the segmenter's fallback when a
result carries no usable word timings, and the sidecar omitted it entirely.
"""
import importlib.util
from pathlib import Path

import pytest

from services.segmentation import segment_transcript

SIDECAR = (
    Path(__file__).resolve().parents[1]
    / "engines" / "indicconformer_nepali" / "main.py"
)


def _load_sidecar():
    """Import the sidecar module directly.

    It is written to run under the engine's OWN venv (AI4Bharat NeMo), so it is
    never importable as a package from the parent venv. Only the pure-stdlib
    result-shaping helper is exercised here — nothing that touches NeMo.
    """
    spec = importlib.util.spec_from_file_location("_indic_sidecar", SIDECAR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Hypothesis:
    def __init__(self, text):
        self.text = text


def test_result_carries_the_real_duration_and_top_level_text():
    sidecar = _load_sidecar()
    result = sidecar._normalize_result([_Hypothesis("नमस्ते संसार")], 11.0)

    assert result["segments"][0]["end"] == 11.0
    assert result["chunks"][0]["timestamp"] == (0.0, 11.0)
    # The segmenter's no-word-timings fallback reads this key.
    assert result["text"] == "नमस्ते संसार"
    assert result["language"] == "ne"


def test_transcript_spans_the_clip_instead_of_its_first_tenth_of_a_second():
    sidecar = _load_sidecar()
    duration = 11.0
    text = "नमस्ते संसार यो नेपाली भाषाको परीक्षण हो"
    result = sidecar._normalize_result([_Hypothesis(text)], duration)

    segments = segment_transcript(result, duration=duration)

    assert segments, "a non-empty transcript must produce at least one segment"
    # The bug pinned this at 0.1 regardless of how long the audio really was.
    assert segments[-1]["end"] == pytest.approx(duration, abs=0.5)
    assert segments[-1]["end"] > 1.0


def test_empty_hypothesis_stays_empty():
    sidecar = _load_sidecar()
    result = sidecar._normalize_result([], 11.0)

    assert result["segments"] == []
    assert result["chunks"] == []
    assert result["text"] == ""
    assert segment_transcript(result, duration=11.0) == []


def test_zero_duration_audio_does_not_claim_a_bogus_end():
    """A clip we could not measure must not assert a zero-length segment."""
    sidecar = _load_sidecar()
    result = sidecar._normalize_result([_Hypothesis("नमस्ते")], 0.0)

    assert result["segments"][0]["end"] is None
