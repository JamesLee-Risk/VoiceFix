"""Reading sources into float32 memory-mapped arrays and writing exports.

Audio is always shaped (frames, channels) float32 and kept at its native rate.
Large files are streamed block by block so memory stays bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import soundfile as sf
import soxr

SUPPORTED_INPUT_SUFFIXES = (".wav", ".flac", ".aiff", ".aif", ".mp3")
READ_BLOCK_FRAMES = 1 << 20


@dataclass(frozen=True)
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    subtype: str
    format: str

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    @property
    def bit_depth_label(self) -> str:
        return {
            "PCM_16": "16-bit",
            "PCM_24": "24-bit",
            "PCM_32": "32-bit",
            "FLOAT": "32-bit 浮點",
            "DOUBLE": "64-bit 浮點",
            "PCM_U8": "8-bit",
            "PCM_S8": "8-bit",
        }.get(self.subtype, self.subtype)


class AudioFormatError(ValueError):
    pass


def probe(path: str | Path) -> AudioInfo:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        raise AudioFormatError(f"不支援的檔案格式：{path.suffix}（支援 WAV、FLAC、AIFF、MP3）")
    try:
        info = sf.info(str(path))
    except RuntimeError as exc:
        raise AudioFormatError(f"無法讀取音檔：{path.name}（{exc}）") from exc
    if info.channels not in (1, 2):
        raise AudioFormatError(f"只支援單聲道或立體聲，此檔案有 {info.channels} 個聲道")
    return AudioInfo(str(path), info.samplerate, info.channels, info.frames, info.subtype, info.format)


def decode_blocks(
    path: str | Path, frames: int, progress: Optional[Callable[[float], None]] = None
) -> Iterator[np.ndarray]:
    """Yield exactly ``frames`` frames in blocks; MP3 frame counts can be off by a
    little, so the tail is trimmed or zero-padded."""
    written = 0
    with sf.SoundFile(str(path)) as f:
        channels = f.channels
        for block in f.blocks(blocksize=READ_BLOCK_FRAMES, dtype="float32", always_2d=True):
            block = block[: frames - written]
            written += len(block)
            yield block
            if progress:
                progress(written / max(frames, 1))
            if written >= frames:
                return
    if written < frames:
        yield np.zeros((frames - written, channels), dtype=np.float32)


@dataclass(frozen=True)
class ExportFormat:
    key: str
    label: str
    suffix: str
    format: str
    subtype: str


EXPORT_FORMATS = (
    ExportFormat("wav24", "WAV 24-bit", ".wav", "WAV", "PCM_24"),
    ExportFormat("wav16", "WAV 16-bit", ".wav", "WAV", "PCM_16"),
    ExportFormat("flac", "FLAC 24-bit", ".flac", "FLAC", "PCM_24"),
    ExportFormat("mp3", "MP3", ".mp3", "MP3", "MPEG_LAYER_III"),
)


def export_format(key: str) -> ExportFormat:
    for fmt in EXPORT_FORMATS:
        if fmt.key == key:
            return fmt
    raise ValueError(f"未知的匯出格式：{key}")


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sr: int,
    fmt: ExportFormat,
    progress: Optional[Callable[[float], None]] = None,
) -> None:
    """Write in blocks (works on memmaps without loading everything)."""
    total = len(audio)
    blocks = (audio[start : start + READ_BLOCK_FRAMES] for start in range(0, total, READ_BLOCK_FRAMES))
    write_blocks(path, blocks, total, audio.shape[1], sr, fmt, progress)


def write_blocks(
    path: str | Path,
    blocks: Iterable[np.ndarray],
    total: int,
    channels: int,
    sr: int,
    fmt: ExportFormat,
    progress: Optional[Callable[[float], None]] = None,
) -> None:
    """Write a stream of (frames, channels) blocks, clipped to full scale."""
    written = 0
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=channels, format=fmt.format, subtype=fmt.subtype) as f:
        for block in blocks:
            f.write(np.clip(np.asarray(block, dtype=np.float32), -1.0, 1.0))
            written += len(block)
            if progress:
                progress(written / max(total, 1))


def _to_channels(block: np.ndarray, channels: int) -> np.ndarray:
    if block.shape[1] == channels:
        return block
    if block.shape[1] == 1:
        return np.repeat(block, channels, axis=1)
    return block.mean(axis=1, keepdims=True)


def iter_resampled(path: str | Path, sr: int, frames: int, channels: int) -> Iterator[np.ndarray]:
    """Stream a file at ``sr`` with ``channels`` channels, exactly ``frames`` long
    (trimmed or zero-padded). Used for accompaniments, so memory stays bounded.
    libsoxr's streaming resampler keeps state between blocks, so joins are seamless."""
    emitted = 0
    with sf.SoundFile(str(path)) as f:
        stream = None if f.samplerate == sr else soxr.ResampleStream(f.samplerate, sr, f.channels, dtype="float32", quality="VHQ")
        blocks = f.blocks(blocksize=READ_BLOCK_FRAMES, dtype="float32", always_2d=True)
        for block, last in _with_last(blocks):
            out = block if stream is None else stream.resample_chunk(block, last=last)
            piece = _to_channels(np.asarray(out, dtype=np.float32), channels)[: frames - emitted]
            emitted += len(piece)
            if len(piece):
                yield piece
            if emitted >= frames:
                return
    if emitted < frames:
        yield np.zeros((frames - emitted, channels), dtype=np.float32)


def _with_last(items: Iterable[np.ndarray]) -> Iterator[tuple[np.ndarray, bool]]:
    iterator = iter(items)
    previous = next(iterator, None)
    for item in iterator:
        yield previous, False
        previous = item
    if previous is not None:
        yield previous, True
