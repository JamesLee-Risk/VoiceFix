"""Breath detection: noisy (spectrally flat), clearly above the noise floor but
well below the surrounding singing, and between 0.12 and 1 second long.

Every threshold is relative to a local +-1.5 s neighbourhood, so a padded
chunk gives the same answer as the whole file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from voicefix.core.analysis import runs_above
from voicefix.core.regions import Region

FRAME_SECONDS = 0.01
NEIGHBOURHOOD_FRAMES = 301  # 3 s
FLAT_BAND_HZ = (300.0, 10_000.0)
EVENT_PAD_SECONDS = 0.02


@dataclass(frozen=True)
class BreathSettings:
    flatness: float = 0.25  # 0 = pure tone, 1 = white noise
    below_peak_db: float = 14.0
    above_floor_db: float = 8.0
    min_seconds: float = 0.12
    max_seconds: float = 1.0


def _frames(mono: np.ndarray, sr: int, offset: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(level_db, flatness, frame, lead) per 10 ms frame aligned to absolute positions."""
    frame = int(round(sr * FRAME_SECONDS))
    lead = offset % frame
    data = np.pad(mono, (lead, (-(len(mono) + lead)) % frame))
    n = len(data) // frame
    level = 10.0 * np.log10(np.maximum((data.reshape(n, frame) ** 2).mean(axis=1), 1e-12))
    nfft = 1 << int(np.ceil(np.log2(frame * 2)))
    freqs, _, spec = signal.stft(data, fs=sr, nperseg=nfft, noverlap=nfft - frame, boundary="even", padded=True)
    power = np.abs(spec[(freqs >= FLAT_BAND_HZ[0]) & (freqs <= min(FLAT_BAND_HZ[1], sr * 0.45))]) ** 2 + 1e-20
    flatness = np.exp(np.log(power).mean(axis=0)) / power.mean(axis=0)
    return level, flatness[:n] if len(flatness) >= n else np.pad(flatness, (0, n - len(flatness))), frame, lead


def detect_breaths(audio: np.ndarray, sr: int, settings: BreathSettings, offset: int = 0) -> list[Region]:
    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    if len(mono) < sr // 2:
        return []
    level, flatness, frame, lead = _frames(mono, sr, offset)
    peak = maximum_filter1d(level, NEIGHBOURHOOD_FRAMES, mode="nearest")
    floor = minimum_filter1d(uniform_filter1d(level, 5, mode="nearest"), NEIGHBOURHOOD_FRAMES, mode="nearest")
    hot = (
        (flatness > settings.flatness)
        & (level > floor + settings.above_floor_db)
        & (level < peak - settings.below_peak_db)
    )
    runs = []
    for lo, hi in runs_above(hot):
        if runs and lo - runs[-1][1] <= 3:
            runs[-1] = (runs[-1][0], hi)
        else:
            runs.append((lo, hi))
    lo_frames, hi_frames = settings.min_seconds / FRAME_SECONDS, settings.max_seconds / FRAME_SECONDS
    pad = int(EVENT_PAD_SECONDS * sr)
    base = offset - lead
    return [
        Region(max(offset, base + lo * frame - pad), min(offset + len(mono), base + hi * frame + pad))
        for lo, hi in runs
        if lo_frames <= hi - lo <= hi_frames
    ]
