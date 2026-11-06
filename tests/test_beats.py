"""Beat tracking on an accompaniment and vocal onset detection."""

import numpy as np
import pytest

from tests.conftest import SR, noise
from voicefix.dsp import beats

BPM = 100.0
BEAT = 60.0 / BPM


def drum_loop(seconds: float, bpm: float = BPM, drift: float = 0.0) -> np.ndarray:
    """Kick on every beat, hats on eighths, a pad underneath; optional tempo drift."""
    n = int(seconds * SR)
    out = np.zeros(n, np.float64)
    rng = np.random.default_rng(3)
    t, k = 0.3, 0
    while t < seconds - 0.2:
        bpm_now = bpm * (1 + drift * k / 100)
        start = int(t * SR)
        kick_t = np.arange(int(0.12 * SR)) / SR
        kick = np.sin(2 * np.pi * 60 * kick_t) * np.exp(-kick_t * 25) * 0.6
        out[start : start + len(kick)] += kick[: n - start]
        for half in (0, 0.5):
            h = int((t + half * 60 / bpm_now) * SR)
            hat = rng.standard_normal(int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / 200) * 0.15
            out[h : h + len(hat)] += hat[: max(0, n - h)]
        k += 1
        t += 60.0 / bpm_now
    tt = np.arange(n) / SR
    out += 0.03 * np.sin(2 * np.pi * 220 * tt)
    return np.stack([out, out], axis=1).astype(np.float32) + noise(n, -70)


def test_tempo_and_beat_positions():
    grid = beats.track_beats(drum_loop(20.0), SR)
    assert grid.bpm == pytest.approx(BPM, rel=0.04)
    assert grid.is_clear
    expected = 0.3 + BEAT * np.arange(5, 25)
    found = grid.beats / SR
    for e in expected:
        assert np.min(np.abs(found - e)) < 0.02


def test_follows_tempo_drift():
    grid = beats.track_beats(drum_loop(30.0, drift=0.3), SR)  # speeds up ~10 % over the song
    intervals = np.diff(grid.beats) / SR
    assert intervals[-5:].mean() < intervals[:5].mean() * 0.97


def test_free_tempo_pad_is_not_clear():
    t = np.arange(15 * SR) / SR
    pad = (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.13 * t))).astype(np.float32)
    grid = beats.track_beats(np.stack([pad, pad], axis=1) + noise(len(pad), -60), SR)
    assert not grid.is_clear


def test_subdivide_and_deviations():
    grid = beats.BeatGrid(np.array([0, 1000, 2000]), 120.0, 3.0)
    sixteenths = grid.subdivide(4)
    assert sixteenths.tolist() == [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    dev, spacing = beats.deviations(np.array([260, 990, 1620]), sixteenths)
    assert dev.tolist() == [10, -10, 120] and spacing.tolist() == [250, 250, 250]


def test_vocal_onsets_found_at_syllable_starts():
    starts = [0.5, 1.1, 1.75, 2.4]
    n = 3 * SR
    mono = np.zeros(n, np.float32)
    for s in starts:
        t = np.arange(int(0.4 * SR)) / SR
        mono[int(s * SR) : int(s * SR) + len(t)] += (0.2 * np.sin(2 * np.pi * 300 * t) * np.minimum(1, t / 0.005)).astype(np.float32)
    onsets = beats.vocal_onsets(np.stack([mono, mono], axis=1) + noise(n, -65), SR) / SR
    assert len(onsets) == len(starts)
    assert np.allclose(onsets, starts, atol=0.01)
