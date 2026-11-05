"""Gain envelopes and a linked look-ahead true-peak limiter.

The limiter is built from local min/box filters only (no recursion), so a chunk
processed with enough padding is identical to processing the whole file.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from voicefix.core.regions import Region, region_mask
from voicefix.dsp.filters import bandpass_sos, highpass_sos, zero_phase

LIMITER_WINDOW_SECONDS = 0.05
LIMITER_MARGIN_DB = 0.3
OVERSAMPLE = 4


def segment_gain_envelope(
    n: int, offset: int, gains: Sequence[tuple[Region, float]], fade: int
) -> np.ndarray:
    """Linear gain per sample: each region gets its dB gain, ramped at the edges."""
    env_db = np.zeros(n, dtype=np.float32)
    for region, gain_db in gains:
        if gain_db == 0.0 or region.end <= offset or region.start >= offset + n:
            continue
        env_db += region_mask(n, [region], fade, fades_inside=True, offset=offset) * np.float32(gain_db)
    return (10.0 ** (env_db / 20.0)).astype(np.float32)


def linked_true_peak_envelope(audio: np.ndarray) -> np.ndarray:
    """Per-sample max of |x| over channels and the 4x oversampled signal."""
    data = np.asarray(audio, dtype=np.float64)
    up = np.abs(signal.resample_poly(data, OVERSAMPLE, 1, axis=0)).max(axis=1)
    up = up[: len(data) * OVERSAMPLE].reshape(len(data), OVERSAMPLE).max(axis=1)
    return np.maximum(up, np.abs(data).max(axis=1))


ENV_FRAME_SECONDS = 0.001


def frame_power_aligned(mono: np.ndarray, sr: int, offset: int) -> tuple[np.ndarray, int, int]:
    """Mean power per 1 ms frame on a grid aligned to absolute sample positions.

    Returns (power, frame, lead) where ``lead`` zeros were prepended for alignment.
    """
    frame = max(1, int(round(sr * ENV_FRAME_SECONDS)))
    lead = offset % frame
    data = np.pad(np.asarray(mono, dtype=np.float64), (lead, (-(len(mono) + lead)) % frame))
    return (data**2).reshape(-1, frame).mean(axis=1), frame, lead


def frames_to_samples(values: np.ndarray, frame: int, lead: int, n: int) -> np.ndarray:
    """Linearly interpolate per-frame values to per-sample values."""
    centres = np.arange(len(values)) * frame + frame / 2.0 - lead
    return np.interp(np.arange(n), centres, values).astype(np.float32)


def power_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def soft_knee_reduction(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Gain reduction in dB (<= 0) of a downward compressor with a soft knee."""
    over = level_db - threshold_db
    slope = 1.0 - 1.0 / ratio
    half = knee_db / 2.0
    in_knee = -slope * (over + half) ** 2 / (2.0 * max(knee_db, 1e-9))
    return np.where(over <= -half, 0.0, np.where(over >= half, -slope * over, in_knee))


def compress(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    makeup_db: float,
    offset: int = 0,
    rms_seconds: float = 0.02,
    smooth_seconds: float = 0.05,
    knee_db: float = 6.0,
) -> np.ndarray:
    """Linked RMS compressor with symmetric (look-ahead) smoothing; no recursion,
    so padded chunks reproduce whole-file processing."""
    if len(audio) == 0:
        return audio
    power, frame, lead = frame_power_aligned(np.asarray(audio).mean(axis=1), sr, offset)
    rms_frames = max(1, int(rms_seconds / ENV_FRAME_SECONDS))
    level = power_db(uniform_filter1d(power, rms_frames, mode="nearest"))
    reduction = soft_knee_reduction(level, threshold_db, ratio, knee_db)
    smoothed = uniform_filter1d(reduction, max(1, int(smooth_seconds / ENV_FRAME_SECONDS)), mode="nearest")
    gain = 10.0 ** ((frames_to_samples(smoothed, frame, lead, len(audio)) + makeup_db) / 20.0)
    return (np.asarray(audio, dtype=np.float32) * gain[:, None]).astype(np.float32)


DEESS_SPLIT_HZ = 4000.0
DEESS_BAND = (4000.0, 10000.0)
HF_DOMINANCE_DB = -15.0  # sibilants carry most of their energy in the HF band


def sibilance_levels(audio: np.ndarray, sr: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(hf_band_db, full_db, frame, lead) per 1 ms frame, 5 ms RMS."""
    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    band = zero_phase(bandpass_sos(sr, *DEESS_BAND), mono[:, None])[:, 0]
    hf_power, frame, lead = frame_power_aligned(band, sr, offset)
    full_power, _, _ = frame_power_aligned(mono, sr, offset)
    window = max(1, int(0.005 / ENV_FRAME_SECONDS))
    hf_db = power_db(uniform_filter1d(hf_power, window, mode="nearest"))
    full_db = power_db(uniform_filter1d(full_power, window, mode="nearest"))
    return hf_db, full_db, frame, lead


def deess(
    audio: np.ndarray, sr: int, threshold_db: float, max_reduction_db: float, offset: int = 0
) -> np.ndarray:
    """Split-band de-esser: only the band above DEESS_SPLIT_HZ is turned down."""
    if len(audio) == 0:
        return audio
    hf_db, full_db, frame, lead = sibilance_levels(audio, sr, offset)
    dominance = np.clip((hf_db - full_db - HF_DOMINANCE_DB) / 6.0, 0.0, 1.0)
    reduction = -np.clip((hf_db - threshold_db) * 0.75, 0.0, max_reduction_db) * dominance
    reduction = uniform_filter1d(reduction, max(1, int(0.003 / ENV_FRAME_SECONDS)), mode="nearest")
    gain = 10.0 ** (frames_to_samples(reduction, frame, lead, len(audio)) / 20.0)
    high = zero_phase(highpass_sos(sr, DEESS_SPLIT_HZ, 2), audio)
    low = np.asarray(audio, dtype=np.float32) - high
    return (low + high * gain[:, None]).astype(np.float32)


def limit(audio: np.ndarray, sr: int, ceiling_db: float) -> np.ndarray:
    if len(audio) == 0:
        return audio
    ceiling = 10.0 ** ((ceiling_db - LIMITER_MARGIN_DB) / 20.0)
    peak = linked_true_peak_envelope(audio)
    target = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if target.min() >= 1.0:
        return np.asarray(audio, dtype=np.float32)
    width = 2 * int(LIMITER_WINDOW_SECONDS * sr / 2) + 1
    held = minimum_filter1d(target, width, mode="nearest")
    gain = np.minimum(uniform_filter1d(held, width, mode="nearest"), target)
    return (np.asarray(audio, dtype=np.float64) * gain[:, None]).astype(np.float32)
