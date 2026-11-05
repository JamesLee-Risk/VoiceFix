"""Sample-accurate regions and the masks that decide where processing applies.

Convention: a *keep mask* is 1.0 where the original signal must be kept and 0.0
where the processed signal is used. Fades always sit on the processed side of a
boundary, so every area that must stay untouched (outside a selection, inside a
protected region) is reproduced sample-for-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

DEFAULT_FADE_SECONDS = 0.008  # 5-10 ms per PRD


@dataclass(frozen=True, order=True)
class Region:
    start: int  # first sample, inclusive
    end: int  # last sample, exclusive

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Region end {self.end} < start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Region") -> bool:
        return self.start < other.end and other.start < self.end

    def to_json(self) -> list[int]:
        return [self.start, self.end]

    @staticmethod
    def from_json(data: Sequence[int]) -> "Region":
        return Region(int(data[0]), int(data[1]))


class ScopeMode(str, Enum):
    WHOLE = "whole"
    SELECTION = "selection"
    EXCLUDE_SELECTION = "exclude_selection"
    EVENTS = "events"


SCOPE_LABELS = {
    ScopeMode.WHOLE: "整個檔案",
    ScopeMode.SELECTION: "只套用選取區段",
    ScopeMode.EXCLUDE_SELECTION: "整個檔案但排除選取區段",
    ScopeMode.EVENTS: "偵測到的事件",
}


def merge_regions(regions: Iterable[Region]) -> list[Region]:
    """Sort and merge overlapping or touching regions; drop empty ones."""
    merged: list[Region] = []
    for region in sorted(r for r in regions if r.length > 0):
        if merged and region.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Region(last.start, max(last.end, region.end))
        else:
            merged.append(region)
    return merged


def clip_regions(regions: Iterable[Region], start: int, end: int) -> list[Region]:
    """Regions intersected with [start, end), in absolute sample positions."""
    clipped = (Region(max(r.start, start), min(r.end, end)) for r in regions if r.end > start and r.start < end)
    return [r for r in clipped if r.length > 0]


def _raised_cosine(length: int) -> np.ndarray:
    """Rising ramp 0 -> 1 (exclusive of both endpoints) of the given length."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    t = (np.arange(length, dtype=np.float64) + 1.0) / (length + 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def region_mask(
    n: int,
    regions: Iterable[Region],
    fade: int,
    fades_inside: bool,
    offset: int = 0,
) -> np.ndarray:
    """Mask of length ``n`` covering absolute samples [offset, offset + n).

    The mask is exactly 1.0 inside each region. With ``fades_inside`` the ramps
    lie within the region edges (the region interior is then not fully 1.0 near
    its edges); otherwise the ramps extend outward and the region is exactly 1.0.
    """
    mask = np.zeros(n, dtype=np.float32)
    ramp = _raised_cosine(fade)
    for region in merge_regions(regions):
        lo, hi = region.start - offset, region.end - offset
        if hi + fade <= 0 or lo - fade >= n:
            continue
        if fades_inside:
            body_lo, body_hi = lo + fade, hi - fade
            rise_lo, fall_hi = lo, hi
        else:
            body_lo, body_hi = lo, hi
            rise_lo, fall_hi = lo - fade, hi + fade
        if body_hi > body_lo:
            mask[max(body_lo, 0) : max(min(body_hi, n), 0)] = 1.0
        else:  # region shorter than two fades: use a centred, shortened ramp
            body_lo = body_hi = (lo + hi) // 2
        _write_max(mask, rise_lo, ramp[: max(body_lo - rise_lo, 0)])
        _write_max(mask, body_hi, ramp[: max(fall_hi - body_hi, 0)][::-1])
    return mask


def _write_max(mask: np.ndarray, at: int, values: np.ndarray) -> None:
    lo, hi = at, at + len(values)
    src_lo, src_hi = max(0, -lo), len(values) - max(0, hi - len(mask))
    if src_hi <= src_lo:
        return
    dst = slice(lo + src_lo, lo + src_hi)
    mask[dst] = np.maximum(mask[dst], values[src_lo:src_hi])


def keep_mask(
    n: int,
    offset: int,
    scope: ScopeMode,
    selections: Sequence[Region],
    protected: Sequence[Region],
    fade: int,
) -> np.ndarray:
    """Where the original signal is kept, given scope and protected regions."""
    if scope == ScopeMode.SELECTION:
        keep = 1.0 - region_mask(n, selections, fade, fades_inside=True, offset=offset)
    elif scope == ScopeMode.EXCLUDE_SELECTION:
        keep = region_mask(n, selections, fade, fades_inside=False, offset=offset)
    else:
        keep = np.zeros(n, dtype=np.float32)
    if protected:
        keep = np.maximum(keep, region_mask(n, protected, fade, fades_inside=False, offset=offset))
    return keep


def blend(original: np.ndarray, processed: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Mix per sample; samples with keep == 1 are copied from ``original`` exactly."""
    if not np.any(keep):
        return processed
    k = keep[:, None]
    mixed = original * k + processed * (1.0 - k)
    return np.where(k >= 1.0, original, mixed).astype(np.float32, copy=False)


def fade_samples(sr: int, seconds: float = DEFAULT_FADE_SECONDS) -> int:
    return max(1, int(round(sr * seconds)))
