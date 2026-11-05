"""Disk cache of step outputs as memory-mapped .npy files.

Entries are content-addressed (the key hashes the input key and parameters), so
an unchanged prefix of the chain is never recomputed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Optional

import numpy as np

from voicefix.core import audio_io

log = logging.getLogger(__name__)
PARTIAL_SUFFIX = ".partial"


def make_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


class AudioCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for leftover in self.directory.glob(f"*{PARTIAL_SUFFIX}"):  # from a crash or kill
            self._remove(leftover)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.npy"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Optional[np.ndarray]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:  # truncated by a crash: drop it
            log.warning("Discarding unreadable cache entry %s: %s", path.name, exc)
            self._remove(path)
            return None

    @contextmanager
    def writer(self, key: str, shape: tuple[int, int]) -> Iterator["SequentialWriter"]:
        """Yield an append-only writer; the entry becomes visible under ``key`` only
        when exactly ``shape[0]`` frames were written."""
        final = self._path(key)
        tmp = self.directory / f"{key}.{uuid.uuid4().hex}{PARTIAL_SUFFIX}"
        try:
            with open(tmp, "wb") as fh:
                writer = SequentialWriter(fh, shape)
                yield writer
                writer.finish()
        except BaseException:
            self._remove(tmp)
            raise
        try:
            os.replace(tmp, final)
        except OSError:  # another writer won the race and the file is mapped; theirs is identical
            self._remove(tmp)

    def remove(self, key: str) -> None:
        self._remove(self._path(key))

    def prune(self, keep: set[str]) -> None:
        for path in self.directory.glob("*.npy"):
            if path.stem not in keep:
                self._remove(path)

    @staticmethod
    def _remove(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:  # still memory-mapped on Windows; retried on next prune
            pass


class SequentialWriter:
    """Writes a float32 .npy file front to back without memory-mapping it."""

    def __init__(self, fh: BinaryIO, shape: tuple[int, int]) -> None:
        self._fh = fh
        self._shape = shape
        self.written = 0
        np.lib.format.write_array_header_1_0(fh, {"descr": "<f4", "fortran_order": False, "shape": shape})

    def write(self, block: np.ndarray) -> None:
        block = np.ascontiguousarray(block, dtype="<f4")
        if block.ndim != 2 or block.shape[1] != self._shape[1]:
            raise ValueError(f"Block shape {block.shape} does not match {self._shape}")
        if self.written + len(block) > self._shape[0]:
            raise ValueError("Writing past the declared length")
        self._fh.write(block.tobytes())
        self.written += len(block)

    def finish(self) -> None:
        if self.written != self._shape[0]:
            raise ValueError(f"Wrote {self.written} of {self._shape[0]} frames")


def source_key(path: str | Path) -> str:
    stat = Path(path).stat()
    return "src-" + make_key(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)


def load_source(
    path: str | Path, cache: AudioCache, progress: Optional[Callable[[float], None]] = None
) -> tuple[np.ndarray, int, str]:
    """Decode the source once into the cache; returns (audio, sample_rate, key)."""
    info = audio_io.probe(path)
    key = source_key(path)
    cached = cache.get(key)
    if cached is None:
        with cache.writer(key, (info.frames, info.channels)) as out:
            for block in audio_io.decode_blocks(path, info.frames, progress):
                out.write(block)
        cached = cache.get(key)
    return cached, info.sample_rate, key


def load_accompaniment(
    path: str | Path, cache: AudioCache, sr: int, frames: int
) -> tuple[np.ndarray, str]:
    """Stereo accompaniment at ``sr``, aligned to ``frames``, streamed into the cache.

    Returns (memory-mapped audio, cache key).
    """
    audio_io.probe(path)  # readable error for unsupported files
    stat = Path(path).stat()
    key = "acc-" + make_key(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, sr, frames)
    cached = cache.get(key)
    if cached is None:
        with cache.writer(key, (frames, 2)) as out:
            for block in audio_io.iter_resampled(path, sr, frames, 2):
                out.write(block)
        cached = cache.get(key)
    return cached, key
