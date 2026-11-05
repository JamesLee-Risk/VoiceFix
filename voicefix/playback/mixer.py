"""A/B audition: before, after and removed signals on one shared play head.

Switching modes never moves the play head; the old and new signals are
crossfaded for ``CROSSFADE_SECONDS`` so the switch makes no click.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Optional

import numpy as np

from voicefix.core import loudness

CROSSFADE_SECONDS = 0.015  # PRD: 10-20 ms
OUTPUT_CHANNELS = 2


class AuditionMode(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    REMOVED = "removed"


MODE_LABELS = {
    AuditionMode.BEFORE: "處理前",
    AuditionMode.AFTER: "處理後",
    AuditionMode.REMOVED: "被移除的聲音",
}


class AuditionMixer:
    def __init__(self, sr: int) -> None:
        self.sr = sr
        self._lock = threading.Lock()
        self._before = np.zeros((0, 1), dtype=np.float32)
        self._after = self._before
        self._removed = self._before
        self._match_gain = 1.0
        self.gain_match_enabled = True
        self.mode = AuditionMode.AFTER
        self._fade_from: Optional[AuditionMode] = None
        self._fade_pos = 0
        self._fade_len = max(1, int(CROSSFADE_SECONDS * sr))
        self.position = 0
        self.loop = True
        self.offset = 0  # absolute position of buffer sample 0, for display
        self._backing: Optional[np.ndarray] = None  # accompaniment on the absolute timeline
        self.backing_enabled = False

    # ----- buffers ----------------------------------------------------------
    def set_buffers(self, before: np.ndarray, after: np.ndarray, offset: int, keep_position: bool = True) -> None:
        if before.shape != after.shape:
            raise ValueError("before/after buffers must have the same shape")
        gain = loudness.match_gain(before, after, self.sr) if len(before) else 1.0
        with self._lock:
            previous_abs = self.offset + self.position
            self._before, self._after = before, after
            self._removed = (before - after).astype(np.float32)
            self._match_gain = gain
            self.offset = offset
            in_range = 0 <= previous_abs - offset < len(before)
            self.position = previous_abs - offset if keep_position and in_range else 0

    def set_backing(self, backing: Optional[np.ndarray]) -> None:
        """Accompaniment (frames, 2) aligned with the step input; never processed."""
        with self._lock:
            self._backing = backing

    @property
    def has_backing(self) -> bool:
        return self._backing is not None

    @property
    def length(self) -> int:
        return len(self._before)

    @property
    def match_gain_db(self) -> float:
        return loudness.linear_to_db(self._match_gain)

    # ----- control ----------------------------------------------------------
    def set_mode(self, mode: AuditionMode) -> None:
        with self._lock:
            if mode == self.mode:
                return
            self._fade_from = self.mode
            self._fade_pos = 0
            self.mode = mode

    def seek(self, position: int) -> None:
        with self._lock:
            self.position = int(np.clip(position, 0, max(self.length - 1, 0)))

    def signal(self, mode: AuditionMode) -> np.ndarray:
        """The whole buffer as it would be heard in ``mode``."""
        with self._lock:
            return self._signal(mode, 0, self.length, with_backing=False)

    # ----- rendering --------------------------------------------------------
    def _signal(self, mode: AuditionMode, start: int, end: int, with_backing: bool = True) -> np.ndarray:
        if mode == AuditionMode.REMOVED:  # judged on its own, never with the accompaniment
            return self._removed[start:end]
        if mode == AuditionMode.BEFORE:
            vocal = self._before[start:end]
        else:
            after = self._after[start:end]
            vocal = after * np.float32(self._match_gain) if self.gain_match_enabled else after
        if not (with_backing and self.backing_enabled and self._backing is not None):
            return vocal
        backing = np.zeros((end - start, OUTPUT_CHANNELS), dtype=np.float32)
        part = self._backing[self.offset + start : self.offset + end]
        backing[: len(part)] = part
        return vocal + backing

    def _block(self, start: int, end: int) -> np.ndarray:
        out = np.asarray(self._signal(self.mode, start, end), dtype=np.float32)
        if self._fade_from is not None:
            n = end - start
            ramp = np.clip((self._fade_pos + np.arange(n)) / self._fade_len, 0.0, 1.0).astype(np.float32)[:, None]
            out = self._signal(self._fade_from, start, end) * (1.0 - ramp) + out * ramp
            self._fade_pos += n
            if self._fade_pos >= self._fade_len:
                self._fade_from = None
        return out

    def render(self, frames: int) -> tuple[np.ndarray, bool]:
        """Next ``frames`` of output (frames, 2). Returns (block, finished)."""
        result = np.zeros((frames, OUTPUT_CHANNELS), dtype=np.float32)
        with self._lock:
            filled, finished = 0, False
            while filled < frames and self.length:
                end = min(self.length, self.position + frames - filled)
                block = self._block(self.position, end)
                result[filled : filled + len(block)] = block if block.shape[1] == OUTPUT_CHANNELS else block[:, :1]
                filled += len(block)
                self.position = end
                if self.position >= self.length:
                    if not self.loop:
                        finished = True
                        break
                    self.position = 0
            if not self.length:
                finished = True
        return np.clip(result, -1.0, 1.0), finished
