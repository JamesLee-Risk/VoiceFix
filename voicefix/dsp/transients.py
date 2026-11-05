"""Detection and repair of short events: mouth clicks and plosive pops.

There is no mature open-source de-click for speech, so this follows the PRD's
plan: transient detection on a linked (mono-sum) side-chain, then a band-limited
replacement inside the event with short crossfades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from voicefix.core.analysis import runs_above
from voicefix.core.regions import Region, region_mask
from voicefix.dsp.filters import highpass_sos, lowpass_sos, zero_phase

ABSOLUTE_FLOOR_DB = -75.0


def _frame_db(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    power = (x[: n * frame] ** 2).reshape(n, frame).mean(axis=1)
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def _surround_mean(values: np.ndarray, outer: int, inner: int) -> np.ndarray:
    """Mean over a window of ``outer`` frames that excludes the central ``inner`` frames."""
    big = uniform_filter1d(values, outer, mode="nearest") * outer
    small = uniform_filter1d(values, inner, mode="nearest") * inner
    return (big - small) / (outer - inner)


def _runs_to_regions(runs: list[tuple[int, int]], frame: int, pad: int, total: int, offset: int) -> list[Region]:
    return [Region(offset + max(0, lo * frame - pad), offset + min(total, hi * frame + pad)) for lo, hi in runs]


def _merge_close(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for lo, hi in runs:
        if merged and lo - merged[-1][1] <= max_gap:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


@dataclass(frozen=True)
class ClickSettings:
    threshold_db: float  # how far above the surrounding HF level a click must rise
    max_length_ms: float = 8.0
    highpass_hz: float = 2500.0
    repair_lowpass_hz: float = 2000.0


def detect_clicks(audio: np.ndarray, sr: int, settings: ClickSettings, offset: int = 0) -> list[Region]:
    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    if len(mono) < sr // 10:
        return []
    hf = zero_phase(highpass_sos(sr, settings.highpass_hz, 4), mono[:, None])[:, 0]
    frame = max(1, int(sr * 0.0005))
    env = _frame_db(hf, frame)
    bg = _surround_mean(env, outer=81, inner=13)
    hot = (env - bg > settings.threshold_db) & (env > ABSOLUTE_FLOOR_DB)
    runs = _merge_close(runs_above(hot), max_gap=4)
    max_frames = int(settings.max_length_ms / 0.5)
    runs = [r for r in runs if r[1] - r[0] <= max_frames]
    return _runs_to_regions(runs, frame, pad=int(sr * 0.001), total=len(mono), offset=offset)


def repair_clicks(audio: np.ndarray, sr: int, events: list[Region], settings: ClickSettings, offset: int = 0) -> np.ndarray:
    if not events:
        return audio
    smooth = zero_phase(lowpass_sos(sr, settings.repair_lowpass_hz, 4), audio)
    mask = region_mask(len(audio), events, max(1, int(sr * 0.001)), fades_inside=False, offset=offset)[:, None]
    return (audio * (1.0 - mask) + smooth * mask).astype(np.float32)


@dataclass(frozen=True)
class PlosiveSettings:
    threshold_db: float  # rise of the low band above its surroundings
    cutoff_hz: float  # high-pass applied inside each event
    detect_band_hz: float = 150.0
    min_length_ms: float = 10.0
    max_length_ms: float = 200.0


def detect_plosives(audio: np.ndarray, sr: int, settings: PlosiveSettings, offset: int = 0) -> list[Region]:
    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    if len(mono) < sr // 2:
        return []
    low = zero_phase(lowpass_sos(sr, settings.detect_band_hz, 4), mono[:, None])[:, 0]
    frame = max(1, int(sr * 0.005))
    low_db = _frame_db(low, frame)
    full_db = _frame_db(mono, frame)
    bg = _surround_mean(low_db, outer=61, inner=13)
    hot = (low_db - bg > settings.threshold_db) & (low_db - full_db > -6.0) & (low_db > -55.0)
    runs = _merge_close(runs_above(hot), max_gap=2)
    lo_frames, hi_frames = settings.min_length_ms / 5.0, settings.max_length_ms / 5.0
    runs = [r for r in runs if lo_frames <= r[1] - r[0] <= hi_frames]
    return _runs_to_regions(runs, frame, pad=int(sr * 0.01), total=len(mono), offset=offset)


def repair_plosives(
    audio: np.ndarray, sr: int, events: list[Region], settings: PlosiveSettings, offset: int = 0
) -> np.ndarray:
    if not events:
        return audio
    filtered = zero_phase(highpass_sos(sr, settings.cutoff_hz, 2), audio)
    mask = region_mask(len(audio), events, max(1, int(sr * 0.005)), fades_inside=False, offset=offset)[:, None]
    return (audio * (1.0 - mask) + filtered * mask).astype(np.float32)
