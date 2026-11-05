"""ITU-R BS.1770 loudness and true peak, computed chunk by chunk.

K-weighting coefficients follow libebur128, which matches the standard at
48 kHz and adapts the two biquads to other sample rates.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

SILENCE_LUFS = -120.0
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
SUBBLOCK_SECONDS = 0.1  # 400 ms gating blocks, 75 % overlap = 4 sub-blocks
STREAM_CHUNK_SECONDS = 30.0
TRUE_PEAK_OVERSAMPLE = 4


def k_weighting_sos(sr: int) -> np.ndarray:
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return np.array([shelf, highpass], dtype=np.float64)


def power_to_lufs(mean_square: float) -> float:
    if mean_square <= 0.0:
        return SILENCE_LUFS
    return -0.691 + 10.0 * math.log10(mean_square)


def subblock_powers(audio: np.ndarray, sr: int) -> np.ndarray:
    """Channel-summed K-weighted mean square per 100 ms sub-block (streamed)."""
    sos = k_weighting_sos(sr)
    channels = audio.shape[1]
    zi = np.zeros((sos.shape[0], 2, channels))
    sub = int(round(sr * SUBBLOCK_SECONDS))
    chunk = sub * int(STREAM_CHUNK_SECONDS / SUBBLOCK_SECONDS)
    powers: list[np.ndarray] = []
    for start in range(0, len(audio), chunk):
        block = np.asarray(audio[start : start + chunk], dtype=np.float64)
        weighted, zi = signal.sosfilt(sos, block, axis=0, zi=zi)
        full = (len(weighted) // sub) * sub
        if full == 0:
            break
        sq = (weighted[:full] ** 2).reshape(-1, sub, channels).mean(axis=1)
        powers.append(sq.sum(axis=1))
    return np.concatenate(powers) if powers else np.zeros(0)


def integrated_loudness(audio: np.ndarray, sr: int) -> float:
    """Gated integrated loudness in LUFS (BS.1770-4)."""
    sub = subblock_powers(audio, sr)
    if len(sub) < 4:
        return segment_loudness(audio, sr)
    blocks = np.convolve(sub, np.ones(4) / 4.0, mode="valid")
    with np.errstate(divide="ignore"):
        block_lufs = -0.691 + 10.0 * np.log10(blocks)
    gated = blocks[block_lufs > ABSOLUTE_GATE_LUFS]
    if len(gated) == 0:
        return SILENCE_LUFS
    relative = power_to_lufs(float(gated.mean())) + RELATIVE_GATE_LU
    gated = blocks[(block_lufs > ABSOLUTE_GATE_LUFS) & (block_lufs > relative)]
    return power_to_lufs(float(gated.mean())) if len(gated) else SILENCE_LUFS


def segment_loudness(audio: np.ndarray, sr: int) -> float:
    """Ungated K-weighted loudness of a segment, streamed so any length is safe."""
    if len(audio) == 0:
        return SILENCE_LUFS
    sos = k_weighting_sos(sr)
    zi = np.zeros((sos.shape[0], 2, audio.shape[1]))
    chunk = int(STREAM_CHUNK_SECONDS * sr)
    total = 0.0
    for start in range(0, len(audio), chunk):
        weighted, zi = signal.sosfilt(sos, np.asarray(audio[start : start + chunk], dtype=np.float64), axis=0, zi=zi)
        total += float((weighted**2).sum())
    return power_to_lufs(total / len(audio))


def true_peak(audio: np.ndarray) -> float:
    """Linear true peak (4x oversampled), streamed."""
    peak = 0.0
    chunk = 1 << 20
    for start in range(0, len(audio), chunk):
        block = np.asarray(audio[start : start + chunk], dtype=np.float64)
        if len(block) == 0:
            continue
        up = signal.resample_poly(block, TRUE_PEAK_OVERSAMPLE, 1, axis=0)
        peak = max(peak, float(np.max(np.abs(up))), float(np.max(np.abs(block))))
    return peak


def linear_to_db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else -math.inf


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def match_gain(reference: np.ndarray, candidate: np.ndarray, sr: int, max_gain_db: float = 24.0) -> float:
    """Linear gain that brings ``candidate`` to the loudness of ``reference``."""
    ref = integrated_loudness(reference, sr)
    cand = integrated_loudness(candidate, sr)
    if ref <= SILENCE_LUFS or cand <= SILENCE_LUFS:
        return 1.0
    return db_to_linear(float(np.clip(ref - cand, -max_gain_db, max_gain_db)))
