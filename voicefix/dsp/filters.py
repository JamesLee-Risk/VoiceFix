"""Biquad filter design (RBJ cookbook) and zero-phase application.

Zero-phase filtering is used everywhere so no step introduces delay, and the
same filter on every channel keeps binaural timing intact.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import signal


def highpass_sos(sr: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")


def lowpass_sos(sr: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, min(cutoff_hz, sr * 0.45), btype="lowpass", fs=sr, output="sos")


def bandpass_sos(sr: int, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, [low_hz, min(high_hz, sr * 0.45)], btype="bandpass", fs=sr, output="sos")


def peaking_sos(sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b = np.array([1.0 + alpha * a, -2.0 * cos_w0, 1.0 - alpha * a])
    a_coef = np.array([1.0 + alpha / a, -2.0 * cos_w0, 1.0 - alpha / a])
    return np.concatenate([b / a_coef[0], a_coef / a_coef[0]])[None, :]


def shelf_sos(sr: int, f0: float, gain_db: float, high: bool, slope: float = 1.0) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    sq = 2.0 * math.sqrt(a) * alpha
    sign = 1.0 if high else -1.0
    b0 = a * ((a + 1) + sign * (a - 1) * cos_w0 + sq)
    b1 = -2.0 * sign * a * ((a - 1) + sign * (a + 1) * cos_w0)
    b2 = a * ((a + 1) + sign * (a - 1) * cos_w0 - sq)
    a0 = (a + 1) - sign * (a - 1) * cos_w0 + sq
    a1 = 2.0 * sign * ((a - 1) - sign * (a + 1) * cos_w0)
    a2 = (a + 1) - sign * (a - 1) * cos_w0 - sq
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def cascade(sections: Iterable[np.ndarray]) -> np.ndarray:
    parts = [s for s in sections if s is not None and len(s)]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 6))


def zero_phase(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if len(sos) == 0 or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    data = np.asarray(audio, dtype=np.float64)
    padlen = min(len(data) - 1, 3 * (2 * len(sos) + 1))
    return signal.sosfiltfilt(sos, data, axis=0, padlen=max(padlen, 0)).astype(np.float32)
