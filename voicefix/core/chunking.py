"""Fixed-grid chunking with padding, so long files are processed in bounded memory.

The grid always starts at sample 0, which keeps full renders deterministic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from voicefix.core.regions import Region

CHUNK_SECONDS = 30.0
DETECT_PAD_SECONDS = 0.5


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class Chunk:
    core: Region  # samples this chunk is responsible for
    padded: Region  # samples actually read (core plus context on both sides)


def iter_chunks(total: int, chunk: int, pad: int, span: Optional[Region] = None) -> Iterator[Chunk]:
    lo, hi = (span.start, span.end) if span else (0, total)
    grid_start = (lo // chunk) * chunk
    for start in range(grid_start, hi, chunk):
        core = Region(max(start, lo), min(start + chunk, hi))
        if core.length == 0:
            continue
        yield Chunk(core, Region(max(0, core.start - pad), min(total, core.end + pad)))


def check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def detect_chunked(
    audio: np.ndarray,
    sr: int,
    detect: Callable[[np.ndarray, int], list[Region]],
    span: Optional[Region] = None,
    cancel: Optional[threading.Event] = None,
    pad_seconds: float = DETECT_PAD_SECONDS,
) -> list[Region]:
    """Run ``detect(chunk, offset)`` per chunk; keep events that start in the chunk core."""
    found: list[Region] = []
    for c in iter_chunks(len(audio), int(CHUNK_SECONDS * sr), int(pad_seconds * sr), span):
        check_cancel(cancel)
        block = np.asarray(audio[c.padded.start : c.padded.end], dtype=np.float32)
        found.extend(r for r in detect(block, c.padded.start) if c.core.start <= r.start < c.core.end)
    return found
