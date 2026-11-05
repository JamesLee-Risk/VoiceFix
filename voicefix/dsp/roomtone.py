"""Room tone: fill digital silence with the recording's own background sound.

Grains of the noise sample are overlap-added with sine windows (whose squares
sum to one, so the level stays constant). Each grain's source position comes
from a generator seeded by the grain's absolute index, which makes the fill
deterministic and identical whether a file is processed whole or in chunks.
"""

from __future__ import annotations

import numpy as np

from voicefix.core.analysis import frame_power_db, noise_floor_db, runs_above
from voicefix.core.regions import Region

GRAIN_SECONDS = 0.2
SILENCE_BELOW_FLOOR_DB = 15.0  # clearly quieter than the room itself
DIGITAL_SILENCE_DBFS = -90.0
MIN_SILENCE_SECONDS = 0.05
FRAME_SECONDS = 0.01
SEED = 0x5EED


def silent_regions(audio: np.ndarray, sr: int, floor_db: float) -> list[Region]:
    """Stretches far below the room's noise floor (muted or cut spans)."""
    frame_db, frame = frame_power_db(audio, sr, FRAME_SECONDS)
    threshold = max(DIGITAL_SILENCE_DBFS, floor_db - SILENCE_BELOW_FLOOR_DB)
    min_frames = int(MIN_SILENCE_SECONDS / FRAME_SECONDS)
    return [Region(lo * frame, hi * frame) for lo, hi in runs_above(frame_db < threshold) if hi - lo >= min_frames]


def room_floor_db(noise: np.ndarray, sr: int) -> float:
    return noise_floor_db(frame_power_db(noise, sr, FRAME_SECONDS)[0])


def room_tone(noise: np.ndarray, sr: int, start: int, length: int) -> np.ndarray:
    """Room tone for absolute samples [start, start + length)."""
    grain = int(GRAIN_SECONDS * sr)
    hop = grain // 2
    channels = noise.shape[1]
    out = np.zeros((length, channels), dtype=np.float64)
    if len(noise) <= grain or length <= 0:
        return out.astype(np.float32)
    window = np.sin(np.pi * (np.arange(grain) + 0.5) / grain)[:, None]
    # from grain -1 on, so the very first samples of a file also get two
    # overlapping grains and full level
    first, last = max(-1, (start - grain) // hop + 1), (start + length) // hop
    for k in range(first, last + 1):
        pos = k * hop
        src = int(np.random.default_rng((SEED, k + 1)).integers(0, len(noise) - grain))  # seeds must be >= 0
        piece = np.asarray(noise[src : src + grain], dtype=np.float64) * window
        lo, hi = max(pos, start), min(pos + grain, start + length)
        if hi > lo:
            out[lo - start : hi - start] += piece[lo - pos : hi - pos]
    return out.astype(np.float32)
