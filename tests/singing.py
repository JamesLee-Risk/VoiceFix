"""Synthetic sung phrases with breaths and sibilants, for flow B tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from tests.conftest import SR, noise

NOTES_HZ = (220.0, 247.0, 262.0, 294.0, 330.0, 294.0, 262.0, 247.0)


@dataclass(frozen=True)
class Take:
    audio: np.ndarray  # (n, 2)
    notes: list[tuple[int, int]]
    breaths: list[tuple[int, int]]
    sibilants: list[tuple[int, int]]


def _note(freq: float, seconds: float, level: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    f = freq * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))  # vibrato
    phase = 2 * np.pi * np.cumsum(f) / SR
    tone = sum(np.sin(k * phase) / k**1.3 for k in range(1, 25))
    env = np.minimum(1, np.minimum(t / 0.05, (seconds - t) / 0.08))
    return (level * tone / np.max(np.abs(tone)) * env).astype(np.float32)


def _shaped_noise(seconds: float, low: float, high: float, level: float, seed: int) -> np.ndarray:
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=SR, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(n))
    x = x / np.sqrt(np.mean(x**2)) * level
    return (x * np.hanning(n)).astype(np.float32)


def sung_take(phrases: int = 8, seed: int = 0, breath_level: float = 0.012, ess_level: float = 0.08) -> Take:
    """Each phrase: a breath in the gap, then a note that starts with an 's'."""
    pieces, notes, breaths, sibilants = [], [], [], []
    pos = 0

    def add(x: np.ndarray) -> tuple[int, int]:
        nonlocal pos
        pieces.append(x)
        span = (pos, pos + len(x))
        pos += len(x)
        return span

    add(np.zeros(int(0.3 * SR), np.float32))
    for i in range(phrases):
        add(np.zeros(int(0.15 * SR), np.float32))
        breaths.append(add(_shaped_noise(0.35, 400, 6000, breath_level, seed + i)))
        add(np.zeros(int(0.1 * SR), np.float32))
        sibilants.append(add(_shaped_noise(0.12, 5000, 9500, ess_level, seed + 100 + i)))
        notes.append(add(_note(NOTES_HZ[i % len(NOTES_HZ)], 1.4, 0.25)))
    add(np.zeros(int(0.4 * SR), np.float32))
    mono = np.concatenate(pieces)
    audio = np.stack([mono, mono], axis=1) + noise(len(mono), -66, seed=seed + 7)
    return Take(audio.astype(np.float32), notes, breaths, sibilants)
