"""Cheap, streamed signal measurements shared by the step analyzers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicefix.core.regions import Region

STREAM_CHUNK_FRAMES = 1 << 21
FLOOR_DB = -120.0
CLIP_LEVEL = 0.999
MIN_CLIP_RUN = 3


def frame_power_db(audio: np.ndarray, sr: int, frame_seconds: float) -> tuple[np.ndarray, int]:
    """Linked (channel-mean) power per non-overlapping frame, in dBFS.

    Returns (powers_db, frame_length_in_samples).
    """
    frame = max(1, int(round(sr * frame_seconds)))
    chunk = (STREAM_CHUNK_FRAMES // frame) * frame or frame
    parts = []
    for start in range(0, len(audio), chunk):
        block = np.asarray(audio[start : start + chunk], dtype=np.float32)
        n = len(block) // frame
        if n == 0:
            continue
        power = (block[: n * frame] ** 2).mean(axis=1).reshape(n, frame).mean(axis=1)
        parts.append(power)
    power = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(power, 1e-12))
    return db.astype(np.float32), frame


def noise_floor_db(frame_db: np.ndarray) -> float:
    if len(frame_db) == 0:
        return FLOOR_DB
    return float(np.percentile(frame_db, 10))


def quietest_window(frame_db: np.ndarray, frame: int, window_frames: int, skip_digital_silence: bool = True) -> Region | None:
    """The window with the lowest mean power (ignoring pure digital silence)."""
    if len(frame_db) < window_frames or window_frames <= 0:
        return None
    power = 10.0 ** (frame_db.astype(np.float64) / 10.0)
    if skip_digital_silence:
        power = np.where(frame_db <= -100.0, 1.0, power)  # penalise dead-silent frames
    sums = np.convolve(power, np.ones(window_frames), mode="valid")
    best = int(np.argmin(sums))
    return Region(best * frame, (best + window_frames) * frame)


def clipped_runs(audio: np.ndarray) -> list[Region]:
    """Runs of at least MIN_CLIP_RUN samples at full scale, on any channel."""
    found: list[Region] = []
    for start in range(0, len(audio), STREAM_CHUNK_FRAMES):
        block = np.asarray(audio[start : start + STREAM_CHUNK_FRAMES])
        hot = np.any(np.abs(block) >= CLIP_LEVEL, axis=1).astype(np.int8)
        edges = np.diff(np.concatenate(([0], hot, [0])))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if hi - lo >= MIN_CLIP_RUN:
                found.append(Region(start + int(lo), start + int(hi)))
    return found


@dataclass(frozen=True)
class SegmentationSettings:
    frame_seconds: float = 0.01
    min_gap_seconds: float = 0.3
    min_segment_seconds: float = 0.1
    pad_seconds: float = 0.05


def runs_above(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def speech_segments(
    audio: np.ndarray, sr: int, settings: SegmentationSettings = SegmentationSettings()
) -> tuple[list[Region], float, float]:
    """Split into non-silent segments.

    Returns (segments, noise_floor_db, threshold_db).
    """
    frame_db, frame = frame_power_db(audio, sr, settings.frame_seconds)
    if len(frame_db) == 0:
        return [], FLOOR_DB, FLOOR_DB
    floor = noise_floor_db(frame_db)
    loud = float(np.percentile(frame_db, 95))
    threshold = floor + max(6.0, 0.25 * (loud - floor))
    runs = runs_above(frame_db > threshold)

    min_gap = int(settings.min_gap_seconds / settings.frame_seconds)
    merged: list[list[int]] = []
    for lo, hi in runs:
        if merged and lo - merged[-1][1] < min_gap:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])

    min_len = int(settings.min_segment_seconds / settings.frame_seconds)
    pad = int(round(settings.pad_seconds * sr))
    total = len(audio)
    segments = [
        Region(max(0, lo * frame - pad), min(total, hi * frame + pad)) for lo, hi in merged if hi - lo >= min_len
    ]
    return _merge_touching(segments), floor, threshold


def _merge_touching(segments: list[Region]) -> list[Region]:
    out: list[Region] = []
    for seg in segments:
        if out and seg.start <= out[-1].end:
            out[-1] = Region(out[-1].start, max(out[-1].end, seg.end))
        else:
            out.append(seg)
    return out


def quiet_regions(
    frame_db: np.ndarray,
    frame: int,
    above_floor_db: float,
    min_seconds: float,
    sr: int,
    margin_seconds: float = 0.02,
) -> list[Region]:
    """Stretches that stay within ``above_floor_db`` of the noise floor, shrunk by a
    margin so that phrase onsets and tails are never touched."""
    floor = noise_floor_db(frame_db)
    min_frames = int(np.ceil(min_seconds * sr / frame))
    margin = int(margin_seconds * sr)
    out = []
    for lo, hi in runs_above(frame_db < floor + above_floor_db):
        if hi - lo >= min_frames:
            region = Region(lo * frame + margin, hi * frame - margin)
            if region.length > 0:
                out.append(region)
    return out


def spectrum_excerpts(
    audio: np.ndarray, sr: int, nperseg: int, excerpts: int = 8, excerpt_seconds: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """Average power spectrum (Welch) of evenly spaced excerpts of the linked mono signal."""
    from scipy import signal  # local: keeps module import light for the UI

    length = int(excerpt_seconds * sr)
    if len(audio) <= length * excerpts:
        starts = [0]
        length = len(audio)
    else:
        starts = np.linspace(0, len(audio) - length, excerpts).astype(int).tolist()
    mono = np.concatenate([np.asarray(audio[s : s + length], dtype=np.float64).mean(axis=1) for s in starts])
    freqs, power = signal.welch(mono, fs=sr, nperseg=min(nperseg, len(mono)))
    return freqs, power


def gaps_between(segments: list[Region], total: int) -> list[Region]:
    gaps, cursor = [], 0
    for seg in segments:
        if seg.start > cursor:
            gaps.append(Region(cursor, seg.start))
        cursor = max(cursor, seg.end)
    if cursor < total:
        gaps.append(Region(cursor, total))
    return gaps
