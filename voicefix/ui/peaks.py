"""Min/max envelopes for drawing long waveforms at any zoom level."""

from __future__ import annotations

import numpy as np

BASE_BUCKET = 256
STREAM_FRAMES = 1 << 21


class PeakPyramid:
    """Linked (channel-max) min/max per BASE_BUCKET samples, computed once."""

    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.total = len(audio)
        mins, maxs = [], []
        step = (STREAM_FRAMES // BASE_BUCKET) * BASE_BUCKET
        for start in range(0, self.total, step):
            block = np.asarray(audio[start : start + step], dtype=np.float32)
            n = -(-len(block) // BASE_BUCKET)
            padded = np.pad(block, ((0, n * BASE_BUCKET - len(block)), (0, 0)))
            shaped = padded.reshape(n, BASE_BUCKET, -1)
            mins.append(shaped.min(axis=(1, 2)))
            maxs.append(shaped.max(axis=(1, 2)))
        self._min = np.concatenate(mins) if mins else np.zeros(0, np.float32)
        self._max = np.concatenate(maxs) if maxs else np.zeros(0, np.float32)

    def envelope(self, start: int, end: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x_samples, lower, upper) with about ``width`` points for [start, end)."""
        start, end = max(0, int(start)), min(self.total, int(end))
        if end <= start or width <= 0:
            empty = np.zeros(0, np.float32)
            return empty, empty, empty
        per_point = max(1, (end - start) // width)
        if per_point < BASE_BUCKET:
            data = np.asarray(self.audio[start:end], dtype=np.float32)
            return self._reduce(data.min(axis=1), data.max(axis=1), start, per_point, 1)
        lo, hi = start // BASE_BUCKET, -(-end // BASE_BUCKET)
        group = max(1, per_point // BASE_BUCKET)
        return self._reduce(self._min[lo:hi], self._max[lo:hi], lo * BASE_BUCKET, group, BASE_BUCKET)

    @staticmethod
    def _reduce(mins: np.ndarray, maxs: np.ndarray, origin: int, group: int, unit: int):
        n = len(mins) // group
        if n == 0:
            return np.array([origin], np.float32), mins[:1], maxs[:1]
        lower = mins[: n * group].reshape(n, group).min(axis=1)
        upper = maxs[: n * group].reshape(n, group).max(axis=1)
        x = origin + (np.arange(n) * group + group / 2) * unit
        return x.astype(np.float64), lower, upper
