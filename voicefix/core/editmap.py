"""Mapping between the source timeline and the timeline after deletions (A1).

All regions in a project are stored in source-timeline samples so that editing
A1 later never invalidates selections, protected regions or event toggles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

from voicefix.core.regions import Region, merge_regions


@dataclass(frozen=True)
class EditMap:
    deletions: tuple[Region, ...] = ()

    @staticmethod
    def from_deletions(deletions: Iterable[Region]) -> "EditMap":
        return EditMap(tuple(merge_regions(deletions)))

    @property
    def is_identity(self) -> bool:
        return not self.deletions

    def _removed_before(self, src: int) -> int:
        return sum(min(d.end, src) - d.start for d in self.deletions if d.start < src)

    def to_edited(self, src: int) -> int:
        """Source sample -> edited sample. Points inside a deletion collapse to its start."""
        return src - self._removed_before(src)

    def to_source(self, edited: int, prefer_after: bool = True) -> int:
        """Edited sample -> source sample.

        A cut point in the edited timeline corresponds to both ends of the
        deletion; ``prefer_after`` picks the end (use for region starts).
        """
        src = edited
        for d in self.deletions:
            if d.start < src or (prefer_after and d.start == src):
                src += d.length
            else:
                break
        return src

    def region_to_edited(self, region: Region) -> Optional[Region]:
        mapped = Region(self.to_edited(region.start), self.to_edited(region.end))
        return mapped if mapped.length > 0 else None

    def regions_to_edited(self, regions: Iterable[Region]) -> list[Region]:
        mapped = (self.region_to_edited(r) for r in regions)
        return [r for r in mapped if r is not None]

    def region_to_source(self, region: Region) -> Region:
        return Region(self.to_source(region.start, True), self.to_source(region.end, False))

    def edited_length(self, source_length: int) -> int:
        return self.to_edited(source_length)

    def kept_spans(self, source_length: int) -> list[Region]:
        spans, cursor = [], 0
        for d in self.deletions:
            if d.start > cursor:
                spans.append(Region(cursor, min(d.start, source_length)))
            cursor = max(cursor, d.end)
        if cursor < source_length:
            spans.append(Region(cursor, source_length))
        return [s for s in spans if s.length > 0]


COPY_BLOCK_FRAMES = 1 << 20


def _span_gain(span: Region, start: int, end: int, fade: int, fade_in: bool, fade_out: bool) -> np.ndarray:
    """Gain for samples [start, end) of a kept span: linear ramps at cut edges."""
    pos = np.arange(start, end, dtype=np.float64)
    gain = np.ones(end - start, dtype=np.float64)
    if fade_in:
        gain = np.minimum(gain, (pos - span.start) / fade)
    if fade_out:
        gain = np.minimum(gain, (span.end - 1 - pos) / fade)
    return np.clip(gain, 0.0, 1.0).astype(np.float32)


def cut_block(edit_map: EditMap, block: np.ndarray, offset: int, total: int, fade: int) -> Iterator[np.ndarray]:
    """The kept parts of ``block`` (source samples [offset, offset + len)), in order.

    Each join gets a short fade-out / fade-in without overlap, so the output
    length is exact and the timeline map stays valid. Gains depend only on
    absolute positions, so cutting chunk by chunk equals cutting the whole file.
    """
    end = offset + len(block)
    for span in edit_map.kept_spans(total):
        lo, hi = max(span.start, offset), min(span.end, end)
        if hi <= lo:
            continue
        piece = np.array(block[lo - offset : hi - offset], dtype=np.float32)
        f = min(fade, span.length // 2)
        fade_in, fade_out = f > 0 and span.start > 0, f > 0 and span.end < total
        if fade_in or fade_out:
            piece *= _span_gain(span, lo, hi, max(f, 1), fade_in, fade_out)[:, None]
        yield piece


def iter_deletion_blocks(audio: np.ndarray, edit_map: EditMap, fade: int) -> Iterator[np.ndarray]:
    total = len(audio)
    for start in range(0, total, COPY_BLOCK_FRAMES):
        yield from cut_block(edit_map, audio[start : start + COPY_BLOCK_FRAMES], start, total, fade)


def apply_deletions(audio: np.ndarray, edit_map: EditMap, fade: int) -> np.ndarray:
    blocks = list(iter_deletion_blocks(audio, edit_map, fade))
    if not blocks:
        return np.zeros((0, audio.shape[1]), dtype=np.float32)
    return np.concatenate(blocks, axis=0)
