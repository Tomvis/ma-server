"""Tests for the crossfade warmup clamp."""

from __future__ import annotations

import pytest
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.audio import WARMUP_DURATION, _clamped_warmup_size

# 44.1kHz / 16-bit / stereo -> 176400 bytes per second of PCM
PCM_FORMAT = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
UNCLAMPED_WARMUP = int(PCM_FORMAT.pcm_sample_size * WARMUP_DURATION)


def test_warmup_uses_full_duration_when_crossfade_buffer_is_larger() -> None:
    """With room to spare, the warmup is WARMUP_DURATION seconds of PCM."""
    assert _clamped_warmup_size(PCM_FORMAT, UNCLAMPED_WARMUP * 2) == UNCLAMPED_WARMUP


def test_warmup_is_clamped_to_a_smaller_crossfade_buffer() -> None:
    """
    A crossfade buffer smaller than the warmup clamps it.

    Short tracks (or aggressive smart_fades clamping) size the crossfade buffer
    below WARMUP_DURATION seconds. Without the clamp the warmup yields PCM the
    crossfade still needs for fade_out_data, misaligning the end-of-track fade.
    """
    small_buffer = UNCLAMPED_WARMUP // 4
    assert _clamped_warmup_size(PCM_FORMAT, small_buffer) == small_buffer


@pytest.mark.parametrize("crossfade_buffer_size", [0, -1])
def test_warmup_ignores_a_disabled_crossfade_buffer(crossfade_buffer_size: int) -> None:
    """Crossfade disabled (buffer <= 0) leaves the warmup unclamped."""
    assert _clamped_warmup_size(PCM_FORMAT, crossfade_buffer_size) == UNCLAMPED_WARMUP


def test_warmup_never_exceeds_the_crossfade_buffer() -> None:
    """The clamp holds across a sweep of buffer sizes - this is the whole invariant."""
    for crossfade_buffer_size in range(1, UNCLAMPED_WARMUP * 2, UNCLAMPED_WARMUP // 8):
        warmup = _clamped_warmup_size(PCM_FORMAT, crossfade_buffer_size)
        assert warmup <= crossfade_buffer_size
        assert warmup <= UNCLAMPED_WARMUP
        assert warmup > 0
