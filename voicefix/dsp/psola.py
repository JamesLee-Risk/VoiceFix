"""TD-PSOLA pitch shifting with a per-sample pitch ratio.

Used for B6's hard-tune ("effect") style. Unlike a phase vocoder it has no
processing latency, so a pitch target that changes every 10 ms (to flatten
vibrato) lands exactly where intended. Grains are two pitch periods long,
taken from the analysis marks nearest each synthesis mark, and the result is
normalised by the summed window, so a ratio of 1 reproduces the input exactly.
"""

from __future__ import annotations

import numpy as np

UNVOICED_HOP_SECONDS = 0.01


def _marks(f0_at_sample: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Analysis marks one period apart (10 ms apart where unvoiced): (positions, periods)."""
    n = len(f0_at_sample)
    positions, periods = [], []
    t = 0.0
    while t < n:
        f = f0_at_sample[int(t)]
        period = sr / f if f > 0 else sr * UNVOICED_HOP_SECONDS
        positions.append(int(round(t)))
        periods.append(period)
        t += period
    return np.array(positions), np.array(periods)


def psola(audio: np.ndarray, sr: int, f0_at_sample: np.ndarray, ratio_at_sample: np.ndarray) -> np.ndarray:
    """``f0_at_sample`` in Hz (0 = unvoiced), ``ratio_at_sample`` the pitch multiplier."""
    data = np.asarray(audio, dtype=np.float64)
    n = len(data)
    marks, periods = _marks(f0_at_sample, sr)
    out = np.zeros_like(data)
    weight = np.zeros(n)
    ts = float(marks[0]) if len(marks) else float(n)
    while ts < n:
        nearest = int(np.clip(np.searchsorted(marks, ts), 0, len(marks) - 1))
        if nearest > 0 and abs(marks[nearest - 1] - ts) <= abs(marks[nearest] - ts):
            nearest -= 1
        centre, period = int(marks[nearest]), periods[nearest]
        voiced = f0_at_sample[min(centre, n - 1)] > 0
        ratio = ratio_at_sample[min(int(ts), n - 1)] if voiced else 1.0
        half = max(1, int(round(period)))
        window = np.hanning(2 * half + 1)
        src_lo, src_hi = centre - half, centre + half + 1
        dst = int(round(ts)) - centre  # where the grain lands relative to its source
        lo, hi = max(src_lo, -dst, 0), min(src_hi, n - dst, n)
        if hi > lo:
            w = window[lo - src_lo : hi - src_lo]
            out[lo + dst : hi + dst] += data[lo:hi] * w[:, None]
            weight[lo + dst : hi + dst] += w
        ts += period / max(ratio, 1e-3)
    covered = weight > 1e-3
    out[covered] /= weight[covered][:, None]
    out[~covered] = data[~covered]
    return out.astype(np.float32)
