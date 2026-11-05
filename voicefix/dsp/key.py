"""Song key estimate (Krumhansl-Schmuckler profiles on a long-term chroma)."""

from __future__ import annotations

import numpy as np

from voicefix.core.analysis import spectrum_excerpts

NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
CHROMA_RANGE_HZ = (55.0, 2000.0)
KEYS = tuple(f"{n} 大調" for n in NOTE_NAMES) + tuple(f"{n} 小調" for n in NOTE_NAMES)


def chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    freqs, power = spectrum_excerpts(audio, sr, nperseg=16384, excerpts=12)
    keep = (freqs >= CHROMA_RANGE_HZ[0]) & (freqs <= CHROMA_RANGE_HZ[1])
    pitch_class = np.round(12.0 * np.log2(freqs[keep] / 440.0)).astype(int) % 12
    pitch_class = (pitch_class + 9) % 12  # A = 9 when C = 0
    return np.bincount(pitch_class, weights=np.sqrt(power[keep]), minlength=12)


def estimate_key(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """(key name, correlation 0-1). Low correlation means the guess is weak."""
    profile = chroma(audio, sr)
    if not np.any(profile):
        return KEYS[0], 0.0
    scores = []
    for template in (MAJOR_PROFILE, MINOR_PROFILE):
        for tonic in range(12):
            scores.append(float(np.corrcoef(profile, np.roll(template, tonic))[0, 1]))
    best = int(np.argmax(scores))
    return KEYS[best], max(scores[best], 0.0)
