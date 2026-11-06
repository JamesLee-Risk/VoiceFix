"""C7 time alignment: move a harmony / double so its timing follows the lead.

* Offset: harmonies sing other pitches, so waveforms cannot be compared.
  Instead the onset envelopes (rises in level: consonants, phrase starts) are
  cross-correlated in short sliding windows, giving how late or early the
  harmony is at each moment.
* Warp: WSOLA (overlap-add with a small waveform-similarity search) reads the
  harmony at the shifted positions, which moves timing without changing pitch.
  With zero offset it reproduces the input exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FRAME_SECONDS = 0.01
WINDOW_SECONDS = 0.6
HOP_SECONDS = 0.1
MIN_CONFIDENCE = 0.3  # normalised correlation below this: keep the previous estimate
GRAIN_SECONDS = 0.03
SEARCH_SECONDS = 0.004


@dataclass(frozen=True)
class OffsetCurve:
    """Harmony lateness in samples per frame (positive = harmony is late)."""

    lag: np.ndarray
    frame: int

    def at_samples(self, start: int, length: int) -> np.ndarray:
        centres = np.arange(len(self.lag)) * self.frame + self.frame / 2.0
        return np.interp(np.arange(start, start + length), centres, self.lag)


def onset_envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    """Positive changes of the log level per 10 ms frame (pitch independent)."""
    frame = int(round(sr * FRAME_SECONDS))
    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    n = len(mono) // frame
    if n == 0:
        return np.zeros(0)
    level = 10.0 * np.log10(np.maximum((mono[: n * frame].reshape(n, frame) ** 2).mean(axis=1), 1e-10))
    level = np.maximum(level, level.max() - 60.0)  # ignore detail in the noise floor
    return np.maximum(np.diff(level, prepend=level[:1]), 0.0)


def estimate_offsets(lead: np.ndarray, harmony: np.ndarray, sr: int, max_lag_seconds: float) -> OffsetCurve:
    frame = int(round(sr * FRAME_SECONDS))
    a, b = onset_envelope(lead, sr), onset_envelope(harmony, sr)
    n = min(len(a), len(b))
    if n == 0:
        return OffsetCurve(np.zeros(1), frame)
    a, b = a[:n], b[:n]
    window = int(WINDOW_SECONDS / FRAME_SECONDS)
    hop = int(HOP_SECONDS / FRAME_SECONDS)
    max_lag = max(1, int(max_lag_seconds / FRAME_SECONDS))
    centres, lags = [], []
    previous = 0.0
    for start in range(0, max(1, n - window), hop):
        scores = _window_scores(a[start : start + window], b, start, max_lag)
        best = previous
        peak = int(np.argmax(scores))
        if scores[peak] > MIN_CONFIDENCE:
            best = peak - max_lag + _parabolic(scores, peak)
        previous = best
        centres.append(start + window // 2)
        lags.append(best)
    if not lags:
        return OffsetCurve(np.zeros(n), frame)
    per_window = median_filter(np.array(lags), size=5, mode="nearest")
    per_frame = np.interp(np.arange(n), centres, per_window)
    return OffsetCurve(uniform_filter1d(per_frame, hop, mode="nearest") * frame, frame)


def _window_scores(ref: np.ndarray, other: np.ndarray, start: int, max_lag: int) -> np.ndarray:
    """Normalised correlation of ``ref`` with ``other`` at lags -max_lag..max_lag."""
    scores = np.full(2 * max_lag + 1, -1.0)
    ref_energy = np.sqrt((ref**2).sum())
    if ref_energy == 0:
        return scores
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        lo, hi = start + lag, start + lag + len(ref)
        if lo < 0 or hi > len(other):
            continue
        seg = other[lo:hi]
        denom = ref_energy * np.sqrt((seg**2).sum())
        if denom > 0:
            scores[i] = float(ref @ seg) / denom
    return scores


def _parabolic(scores: np.ndarray, peak: int) -> float:
    """Sub-frame offset of the peak from a parabola through its neighbours."""
    if peak == 0 or peak == len(scores) - 1:
        return 0.0
    left, mid, right = scores[peak - 1], scores[peak], scores[peak + 1]
    denom = left - 2 * mid + right
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5)) if denom < 0 else 0.0


def _position(read_position: np.ndarray, k: int) -> float:
    """Read position for output sample k, extended linearly beyond both ends."""
    n = len(read_position)
    if k < 0:
        return read_position[0] + k
    if k >= n:
        return read_position[-1] + (k - n + 1)
    return read_position[k]


def _best_match(data: np.ndarray, natural: int, centre: int, search: int, hop: int) -> int:
    """Start within ``centre`` +- ``search`` whose waveform best continues ``natural``."""
    n = len(data)
    template = data[max(natural, 0) : max(natural, 0) + hop].mean(axis=1)
    lo, hi = max(centre - search, 0), min(centre + search, n - hop)
    if len(template) < hop or hi < lo:
        return centre
    region = data[lo : hi + hop].mean(axis=1)
    scores = np.correlate(region, template, mode="valid")
    return lo + int(np.argmax(scores))


def wsola(audio: np.ndarray, sr: int, read_position: np.ndarray) -> np.ndarray:
    """Output sample t is taken from input position ``read_position[t]`` (pitch kept)."""
    data = np.asarray(audio, dtype=np.float64)
    n = len(data)
    grain = int(GRAIN_SECONDS * sr) // 2 * 2
    hop = grain // 2
    search = int(SEARCH_SECONDS * sr)
    window = np.hanning(grain + 1)[:-1]  # periodic: 50 % overlaps sum to one
    out = np.zeros_like(data)
    weight = np.zeros(n)
    previous_end = None
    for t in range(-hop, n, hop):
        centre = int(round(_position(read_position, t + hop))) - hop
        start = centre
        if previous_end is not None and search > 0:
            natural = previous_end  # the input that would continue the last grain seamlessly
            if abs(natural - centre) <= search:
                start = natural
            else:
                start = _best_match(data, natural, centre, search, hop)
        lo, hi = max(start, 0), min(start + grain, n)
        out_lo = t + (lo - start)
        out_hi = out_lo + (hi - lo)
        clip_lo, clip_hi = max(out_lo, 0), min(out_hi, n)
        if clip_hi > clip_lo:
            w = window[lo - start + (clip_lo - out_lo) : lo - start + (clip_hi - out_lo)]
            src = data[lo + (clip_lo - out_lo) : lo + (clip_hi - out_lo)]
            out[clip_lo:clip_hi] += src * w[:, None]
            weight[clip_lo:clip_hi] += w
        previous_end = start + hop
    covered = weight > 1e-6
    out[covered] /= weight[covered][:, None]
    out[~covered] = data[~covered]
    return out.astype(np.float32)
