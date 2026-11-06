"""Beat grid from the accompaniment, and where the vocal's syllables start.

* Beats: spectral-flux onset strength -> tempo from its autocorrelation ->
  dynamic-programming beat tracking (Ellis 2007), which follows tempo drift.
* Grid: each beat interval split into 2 (eighths) or 4 (sixteenths).
* Vocal onsets: sharp rises of the vocal's level (consonants, syllable starts),
  on a 5 ms grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from voicefix.core.chunking import CHUNK_SECONDS

FRAME_SECONDS = 0.01
TEMPO_RANGE_BPM = (60.0, 200.0)
PREFERRED_BPM = 120.0
TIGHTNESS = 100.0
MAX_PLAUSIBLE_BPM = 140.0
REFINE_SECONDS = 0.025
CLEAR_BEATS = 1.5  # onset strength on beats vs overall; below this the beat is unclear
ONSET_FRAME_SECONDS = 0.005
ONSET_RISE_DB = 9.0  # level rise over 20 ms that counts as a new syllable
ONSET_BELOW_LOUD_DB = 30.0  # ...and it must reach within this of the vocal's loud parts
MIN_ONSET_GAP_SECONDS = 0.12


@dataclass(frozen=True)
class BeatGrid:
    beats: np.ndarray  # sample positions
    bpm: float
    clarity: float  # how much stronger the accompaniment is on its beats

    @property
    def is_clear(self) -> bool:
        return self.clarity >= CLEAR_BEATS and len(self.beats) >= 4

    def subdivide(self, per_beat: int) -> np.ndarray:
        """Grid points: every beat split into ``per_beat`` equal parts."""
        if len(self.beats) < 2:
            return self.beats.astype(np.float64)
        steps = np.arange(per_beat) / per_beat
        inner = (self.beats[:-1, None] + np.diff(self.beats)[:, None] * steps[None, :]).ravel()
        return np.append(inner, self.beats[-1]).astype(np.float64)


def onset_strength(audio: np.ndarray, sr: int) -> np.ndarray:
    """Half-wave rectified spectral flux of the log spectrum per 10 ms frame."""
    hop = int(round(sr * FRAME_SECONDS))
    nfft = 2048 if sr <= 50_000 else 4096
    chunk = int(CHUNK_SECONDS * sr) // hop * hop
    parts, previous = [], None
    for start in range(0, len(audio), chunk):
        mono = np.asarray(audio[start : start + chunk + nfft], dtype=np.float64).mean(axis=1)
        if len(mono) < nfft:
            break
        _, _, spec = signal.stft(mono, fs=sr, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
        log = np.log1p(1000.0 * np.abs(spec))[:, : chunk // hop]
        if previous is not None:
            log = np.concatenate([previous, log], axis=1)
        flux = np.maximum(np.diff(log, axis=1), 0.0).sum(axis=0)
        parts.append(flux if previous is not None else np.concatenate([[0.0], flux]))
        previous = log[:, -1:]
    strength = np.concatenate(parts) if parts else np.zeros(0)
    return strength / (strength.std() + 1e-9)


def estimate_period(strength: np.ndarray) -> float:
    """Beat period in frames from the autocorrelation, leaning towards 120 BPM."""
    lo = int(60.0 / TEMPO_RANGE_BPM[1] / FRAME_SECONDS)
    hi = int(60.0 / TEMPO_RANGE_BPM[0] / FRAME_SECONDS)
    centred = strength - strength.mean()
    corr = signal.correlate(centred, centred, mode="full", method="fft")[len(centred) - 1 :]
    lags = np.arange(lo, min(hi, len(corr) - 1) + 1)
    if len(lags) == 0:
        return 60.0 / PREFERRED_BPM / FRAME_SECONDS
    preferred = 60.0 / PREFERRED_BPM / FRAME_SECONDS
    weight = np.exp(-0.5 * (np.log2(lags / preferred) / 1.0) ** 2)
    best = int(lags[np.argmax(corr[lags] * weight)])
    # octave errors: busy hi-hats make eighths look like beats; prefer the slower
    # level when it is nearly as periodic and the faster one is above MAX_PLAUSIBLE_BPM
    double = 2 * best
    if 60.0 / (best * FRAME_SECONDS) > MAX_PLAUSIBLE_BPM and double < len(corr) and corr[double] >= 0.6 * corr[best]:
        best = double
    return float(best)


def track_beats(accompaniment: np.ndarray, sr: int) -> BeatGrid:
    strength = onset_strength(accompaniment, sr)
    n = len(strength)
    if n < 100:
        return BeatGrid(np.zeros(0), 0.0, 0.0)
    period = estimate_period(strength)
    score = strength.copy()
    back = np.full(n, -1)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    offsets = np.arange(lo, hi + 1)
    penalty = -TIGHTNESS * np.log(offsets / period) ** 2
    for t in range(lo, n):
        prev = t - offsets
        valid = prev >= 0
        if not valid.any():
            continue
        candidates = score[prev[valid]] + penalty[valid]
        best = int(np.argmax(candidates))
        score[t] = strength[t] + candidates[best]
        back[t] = prev[valid][best]
    end = n - int(period) + int(np.argmax(score[n - int(period) :])) if n > period else int(np.argmax(score))
    beats = [end]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    frames = np.array(beats[::-1])
    clarity = float(strength[frames].mean() / (strength.mean() + 1e-9))
    hop = int(round(sr * FRAME_SECONDS))
    centre = (2048 if sr <= 50_000 else 4096) // 2  # flux frame j is centred on j * hop + nfft / 2
    coarse = frames * hop + centre
    return BeatGrid(refine_beats(accompaniment, sr, coarse), 60.0 / (period * FRAME_SECONDS), clarity)


def refine_beats(accompaniment: np.ndarray, sr: int, coarse: np.ndarray) -> np.ndarray:
    """Snap each beat to the steepest level rise within +-REFINE_SECONDS on a 1 ms grid
    (the flux frames are 10 ms, which leaves the attack up to a frame early)."""
    frame = max(1, int(round(sr * 0.001)))
    level = _frame_levels(accompaniment, frame, chunk_frames=int(CHUNK_SECONDS * 1000))
    if len(level) < 4:
        return coarse
    rise = np.concatenate([[0.0, 0.0], level[2:] - level[:-2]])  # over 2 ms
    reach = int(REFINE_SECONDS * 1000)
    refined = []
    for beat in coarse:
        centre = int(beat // frame)
        lo, hi = max(0, centre - reach), min(len(rise), centre + reach + 1)
        refined.append((lo + int(np.argmax(rise[lo:hi]))) * frame if hi > lo else int(beat))
    return np.array(refined)


def vocal_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """Sample positions where the vocal's level jumps (syllable starts)."""
    frame = int(round(sr * ONSET_FRAME_SECONDS))
    level = _frame_levels(audio, frame)
    n = len(level)
    if n < 8:
        return np.zeros(0, dtype=np.int64)
    loud = np.percentile(level, 95)
    rise = level[4:] - level[:-4]  # over 20 ms
    # the level must reach the singing, not just the room: after B2 lowers the
    # gaps, the noise floor coming back at a gap's end is a rise too
    hot = (rise > ONSET_RISE_DB) & (level[4:] > loud - ONSET_BELOW_LOUD_DB)
    starts, last = [], -np.inf
    gap = MIN_ONSET_GAP_SECONDS / ONSET_FRAME_SECONDS
    for i in np.flatnonzero(hot & ~np.concatenate([[False], hot[:-1]])):
        if i - last >= gap:
            starts.append(i)
            last = i
    # the steepest point of each rise marks the onset
    out = []
    for i in starts:
        window = rise[i : i + 4]
        out.append((i + int(np.argmax(window)) + 2) * frame)
    return np.array(out, dtype=np.int64)


def _frame_levels(audio: np.ndarray, frame: int, chunk_frames: int = int(CHUNK_SECONDS / ONSET_FRAME_SECONDS)) -> np.ndarray:
    """Channel-mean level (dB) per frame, read in chunks to keep memory flat."""
    chunk = chunk_frames * frame
    parts = []
    for start in range(0, len(audio) // frame * frame, chunk):
        mono = np.asarray(audio[start : start + chunk], dtype=np.float32).mean(axis=1)
        n = len(mono) // frame
        parts.append((mono[: n * frame].astype(np.float64).reshape(n, frame) ** 2).mean(axis=1))
    power = np.concatenate(parts) if parts else np.zeros(0)
    return 10.0 * np.log10(np.maximum(power, 1e-10))


def deviations(onsets: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(signed distance to the nearest grid point, that grid spacing) per onset;
    positive = late."""
    if len(grid) < 2 or len(onsets) == 0:
        return np.zeros(len(onsets)), np.full(len(onsets), np.inf)
    idx = np.clip(np.searchsorted(grid, onsets), 1, len(grid) - 1)
    left, right = grid[idx - 1], grid[idx]
    nearest = np.where(onsets - left <= right - onsets, left, right)
    return onsets - nearest, right - left
