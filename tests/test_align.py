"""C7 time alignment building blocks."""

import numpy as np
import pytest

from tests.conftest import SR, noise
from voicefix.dsp import align


def phrase_track(onsets: list[float], cents: float, shift_seconds: list[float]) -> np.ndarray:
    """Notes starting at ``onsets`` (+ per-note shift) with a consonant-like burst each; stereo."""
    total = int((max(onsets) + 2.0) * SR)
    mono = np.zeros(total, np.float32)
    rng = np.random.default_rng(int(cents))
    for onset, shift in zip(onsets, shift_seconds):
        start = int((onset + shift) * SR)
        burst = (rng.standard_normal(int(0.04 * SR)) * 0.15 * np.hanning(int(0.04 * SR))).astype(np.float32)
        t = np.arange(int(0.6 * SR)) / SR
        hz = 440.0 * 2 ** ((cents - 6900) / 1200)
        tone = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 8)) * np.minimum(1, (0.6 - t) / 0.05)
        mono[start : start + len(burst)] += burst
        mono[start + len(burst) : start + len(burst) + len(tone)] += (0.2 * tone / 2.6).astype(np.float32)
    return np.stack([mono, mono], axis=1) + noise(total, -65)


ONSETS = [0.5 + 0.9 * k for k in range(12)]
SHIFTS = [-0.03] * 6 + [0.045] * 6


@pytest.fixture(scope="module")
def pair():
    lead = phrase_track(ONSETS, 6900, [0.0] * 12)
    harmony = phrase_track(ONSETS, 7300, SHIFTS)  # a major third up, early then late
    return lead, harmony


def test_offsets_follow_the_harmony_timing(pair):
    lead, harmony = pair
    curve = align.estimate_offsets(lead, harmony, SR, 0.15)
    early = curve.at_samples(int(ONSETS[2] * SR), 1)[0] / SR
    late = curve.at_samples(int(ONSETS[9] * SR), 1)[0] / SR
    assert early == pytest.approx(-0.03, abs=0.012)
    assert late == pytest.approx(0.045, abs=0.012)


def test_wsola_identity_and_pure_delay_are_exact():
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal((SR, 2)) * 0.1).astype(np.float32)
    same = align.wsola(audio, SR, np.arange(len(audio), dtype=np.float64))
    assert np.allclose(same, audio, atol=1e-6)
    shifted = align.wsola(audio, SR, np.arange(len(audio), dtype=np.float64) + 480)
    assert np.allclose(shifted[2000:-3000], audio[2480:-2520], atol=1e-6)


def test_aligned_harmony_lines_up_with_lead(pair):
    lead, harmony = pair
    curve = align.estimate_offsets(lead, harmony, SR, 0.15)
    read = np.arange(len(harmony)) + curve.at_samples(0, len(harmony))
    fixed = align.wsola(harmony, SR, read)
    after = align.estimate_offsets(lead, fixed, SR, 0.15)
    for onset in (ONSETS[2], ONSETS[9]):
        assert abs(after.at_samples(int(onset * SR), 1)[0] / SR) < 0.008
