"""Pitch tracking, note segmentation and pitch shifting for B6.

* Pitch: WORLD's DIO + StoneMask (pyworld) on 10 ms frames.
* Shifting: Rubber Band (via pedalboard) with formant preservation. The R2
  engine is used on purpose: the R3 ("high quality") engine drifts in time by
  about 1 ms per second when shifting, which would smear crossfades and pull
  the vocal away from the accompaniment.
* The hard-tune curve feeds TD-PSOLA (dsp.psola), which has no latency.
* A phase vocoder cannot be cut into chunks without changing the waveform, so
  every note is shifted in one call; identical boundaries make the preview and
  the full render identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pedalboard
import pyworld
from scipy.ndimage import median_filter, uniform_filter1d

from voicefix.core.chunking import CHUNK_SECONDS
from voicefix.core.regions import Region

FRAME_SECONDS = 0.01
F0_RANGE_HZ = (70.0, 1100.0)
TRACK_PAD_SECONDS = 0.5
MIN_NOTE_SECONDS = 0.12
NOTE_JUMP_CENTS = 90.0  # a change this large within 30 ms starts a new note
SHIFT_CONTEXT_SECONDS = 0.05


@dataclass(frozen=True)
class PitchTrack:
    cents: np.ndarray  # per 10 ms frame, MIDI note * 100; NaN where unvoiced
    frame: int  # samples per frame

    def region(self, lo: int, hi: int) -> Region:
        return Region(lo * self.frame, hi * self.frame)


@dataclass(frozen=True)
class Note:
    region: Region
    cents: float  # median pitch of the steady middle part
    frames: tuple[int, int]


def hz_to_cents(hz: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hz > 0, 1200.0 * np.log2(hz / 440.0) + 6900.0, np.nan)


def track_pitch(audio: np.ndarray, sr: int) -> PitchTrack:
    """Streamed over 30 s chunks; frames sit on a 10 ms grid from sample 0."""
    frame = int(round(sr * FRAME_SECONDS))
    chunk = int(CHUNK_SECONDS / FRAME_SECONDS) * frame
    pad = int(TRACK_PAD_SECONDS / FRAME_SECONDS) * frame
    total_frames = len(audio) // frame
    cents = np.full(total_frames, np.nan)
    for start in range(0, total_frames * frame, chunk):
        lo, hi = max(0, start - pad), min(len(audio), start + chunk + pad)
        mono = np.asarray(audio[lo:hi], dtype=np.float64).mean(axis=1)
        f0, times = pyworld.dio(mono, sr, f0_floor=F0_RANGE_HZ[0], f0_ceil=F0_RANGE_HZ[1], frame_period=FRAME_SECONDS * 1000)
        f0 = pyworld.stonemask(mono, f0, times, sr)
        first = (start - lo) // frame
        count = min(chunk // frame, total_frames - start // frame)
        cents[start // frame : start // frame + count] = hz_to_cents(f0[first : first + count])
    return PitchTrack(cents, frame)


def _voiced_runs(cents: np.ndarray) -> list[tuple[int, int]]:
    voiced = ~np.isnan(cents)
    edges = np.diff(np.concatenate(([0], voiced.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segment_notes(track: PitchTrack) -> list[Note]:
    """Split voiced runs at pitch jumps; vibrato (smooth, < 90 cents per 30 ms) stays one note."""
    notes: list[Note] = []
    min_frames = int(MIN_NOTE_SECONDS / FRAME_SECONDS)
    for lo, hi in _voiced_runs(track.cents):
        if hi - lo < min_frames:
            continue
        smooth = median_filter(track.cents[lo:hi], size=7, mode="nearest")
        jumps = np.flatnonzero(np.abs(smooth[3:] - smooth[:-3]) > NOTE_JUMP_CENTS) + 3
        bounds = [0]
        for j in jumps:
            if j - bounds[-1] >= min_frames // 2:
                bounds.append(int(j))
        bounds.append(hi - lo)
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < min_frames:
                continue
            core = smooth[a + (b - a) // 5 : b - (b - a) // 5]  # skip onset scoops and tails
            notes.append(Note(track.region(lo + a, lo + b), float(np.median(core)), (lo + a, lo + b)))
    return notes


def nearest_in_scale(cents: np.ndarray | float, pitch_classes: Sequence[int]) -> np.ndarray:
    """Closest pitch (in cents) whose pitch class is in the scale."""
    values = np.asarray(cents, dtype=np.float64)
    octave = np.floor(values / 1200.0) * 1200.0
    candidates = np.array(sorted(pitch_classes), dtype=np.float64) * 100.0
    options = np.concatenate([candidates - 1200.0, candidates, candidates + 1200.0])
    within = values - octave
    best = options[np.argmin(np.abs(within[..., None] - options), axis=-1)]
    return octave + best


def shift_segment(audio: np.ndarray, sr: int, semitones: np.ndarray | float) -> np.ndarray:
    """Rubber Band R2 with formant preservation; same length, no time drift."""
    data = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).T)
    shift = semitones if np.isscalar(semitones) else np.asarray(semitones, dtype=np.float64)
    out = pedalboard.time_stretch(data, sr, 1.0, shift, high_quality=False, preserve_formants=True)
    out = out[:, : data.shape[1]]
    if out.shape[1] < data.shape[1]:
        out = np.pad(out, ((0, 0), (0, data.shape[1] - out.shape[1])))
    return out.T.astype(np.float32)


def context_region(region: Region, sr: int, total: int) -> Region:
    pad = int(SHIFT_CONTEXT_SECONDS * sr)
    return Region(max(0, region.start - pad), min(total, region.end + pad))


def hard_tune(
    track: PitchTrack, frames: tuple[int, int], target_cents: float, amount: float, retune_frames: int, outer: Region
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (f0 Hz, pitch ratio) over ``outer`` that pulls every frame of
    the note onto ``target_cents``; outside the note the ratio is 1."""
    lo, hi = frames
    cents = track.cents
    shift = np.zeros(len(cents))
    steady = median_filter(np.nan_to_num(cents[lo:hi], nan=target_cents), size=3, mode="nearest")
    shift[lo:hi] = (target_cents - steady) / 100.0 * amount
    size = 2 * (retune_frames // 2) + 1  # odd: an even window shifts the curve half a frame late
    shift = uniform_filter1d(shift, size, mode="nearest")
    index = np.clip((outer.start + np.arange(outer.length)) // track.frame, 0, len(cents) - 1)
    hz = np.where(np.isnan(cents), 0.0, 440.0 * 2.0 ** ((cents - 6900.0) / 1200.0))
    return hz[index], 2.0 ** (shift[index] / 12.0)
