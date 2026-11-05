"""Long-term spectrum analysis for the tone EQ.

Only narrow, persistent resonances are detected automatically. Low-mid
"mud" is deliberately not measured: a singer's fundamental lives in the same
range, so a long-term spectrum cannot tell the two apart. Low mids are left to
the style presets instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicefix.core.analysis import spectrum_excerpts

# Above the second formant and below the sibilant band (B10 handles that).
RESONANCE_RANGE_HZ = (1500.0, 6000.0)
RESONANCE_MIN_DB = 4.0
MAX_RESONANCES = 2
NARROW_FRACTION = 3.0  # 1/3 octave smoothing blurs individual harmonics
TREND_FRACTION = 0.67  # 1.5 octave local trend


@dataclass(frozen=True)
class ToneProfile:
    resonances: tuple[tuple[float, float], ...]  # (hz, dB above the local trend)


def octave_smooth(freqs: np.ndarray, db: np.ndarray, fraction: float) -> np.ndarray:
    """Mean over +- half a 1/fraction octave around each bin."""
    half = 2.0 ** (0.5 / fraction)
    cumulative = np.concatenate(([0.0], np.cumsum(db)))
    lo = np.searchsorted(freqs, freqs / half, side="left")
    hi = np.searchsorted(freqs, freqs * half, side="right")
    return (cumulative[hi] - cumulative[lo]) / np.maximum(hi - lo, 1)


def _peaks(freqs: np.ndarray, excess: np.ndarray, band: tuple[float, float], min_db: float, count: int):
    """Largest values of ``excess`` inside ``band``, at least 1/3 octave apart."""
    inside = np.flatnonzero((freqs >= band[0]) & (freqs <= band[1]))
    found: list[tuple[float, float]] = []
    for i in inside[np.argsort(excess[inside])[::-1]]:
        if excess[i] < min_db or len(found) >= count:
            break
        if all(abs(np.log2(freqs[i] / f)) > 1 / 3 for f, _ in found):
            found.append((float(freqs[i]), float(excess[i])))
    return found


def analyse_tone(audio: np.ndarray, sr: int) -> ToneProfile:
    freqs, power = spectrum_excerpts(audio, sr, nperseg=8192)
    keep = freqs > 20.0
    freqs, db = freqs[keep], 10.0 * np.log10(np.maximum(power[keep], 1e-20))
    excess = octave_smooth(freqs, db, NARROW_FRACTION) - octave_smooth(freqs, db, TREND_FRACTION)
    band = (RESONANCE_RANGE_HZ[0], min(RESONANCE_RANGE_HZ[1], sr * 0.4))
    return ToneProfile(tuple(_peaks(freqs, excess, band, RESONANCE_MIN_DB, MAX_RESONANCES)))
