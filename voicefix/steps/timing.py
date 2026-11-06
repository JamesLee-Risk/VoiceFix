"""Moving a vocal in time without changing its pitch (B7, C7a, C7b).

Shared machinery: phrases (speech segments plus a margin) are warped with
WSOLA one whole phrase at a time, lazily per chunk, and spliced back with short
fades. Whole phrases make a preview identical to the full render.

B7 / C7a: the beat grid comes from the accompaniment; each syllable onset that
is further from the nearest grid point than the tolerance becomes an event
("搶拍 35 ms", "慢 50 ms"), which moves it towards the grid.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from voicefix.core import analysis as an
from voicefix.core.regions import Region, merge_regions, region_mask
from voicefix.dsp import align, beats
from voicefix.steps.base import Analysis, Event, EventStep, Option, Params, ParamSpec, StepContext

PIECE_MARGIN_SECONDS = 0.2  # moved content can reach this far outside its phrase
WARP_CONTEXT_SECONDS = 0.1
PIECE_FADE_SECONDS = 0.025
MAX_PIECE_SECONDS = 15.0
SPLIT_SEARCH_SECONDS = 2.0
MAX_MOVE_SECONDS = 0.15
EVENT_SECONDS = 0.08  # how much of the syllable an event marks on screen
MAX_SLOPE = 0.5  # displacement may change at most this fast, so time never runs backwards


class Displacement(Protocol):
    def at_samples(self, start: int, length: int) -> np.ndarray: ...


@dataclass(frozen=True)
class AnchorCurve:
    """Read offset (samples) interpolated between (output position, offset) anchors."""

    positions: np.ndarray
    offsets: np.ndarray

    def at_samples(self, start: int, length: int) -> np.ndarray:
        if len(self.positions) == 0:
            return np.zeros(length)
        return np.interp(np.arange(start, start + length), self.positions, self.offsets, left=0.0, right=0.0)


class AlignPlan:
    """Pieces warped lazily; a small cache covers pieces that span two chunks."""

    CACHE_SIZE = 4

    def __init__(self, audio: np.ndarray, sr: int, curve: Displacement, amount: float, pieces: list[Region]):
        self.audio, self.sr, self.curve, self.amount, self.pieces = audio, sr, curve, amount, pieces
        self._cache: OrderedDict[Region, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def warped(self, piece: Region) -> np.ndarray:
        with self._lock:
            if piece in self._cache:
                self._cache.move_to_end(piece)
                return self._cache[piece]
            pad = int(WARP_CONTEXT_SECONDS * self.sr)
            outer = Region(max(0, piece.start - pad), min(len(self.audio), piece.end + pad))
            read = np.arange(outer.start, outer.end, dtype=np.float64)
            read += self.curve.at_samples(outer.start, outer.length) * self.amount
            warped = align.wsola(np.asarray(self.audio[outer.start : outer.end]), self.sr, read - outer.start)
            result = warped[piece.start - outer.start : piece.end - outer.start]
            self._cache[piece] = result
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
            return result


def split_long(audio: np.ndarray, sr: int, piece: Region) -> list[Region]:
    """Cut pieces longer than MAX_PIECE_SECONDS at the quietest moment near each cut."""
    limit = int(MAX_PIECE_SECONDS * sr)
    if piece.length <= limit:
        return [piece]
    frame = int(0.01 * sr)
    search = int(SPLIT_SEARCH_SECONDS * sr)
    parts, start = [], piece.start
    while piece.end - start > limit:
        lo, hi = start + limit - search, start + limit
        levels = (np.asarray(audio[lo:hi], dtype=np.float32) ** 2).mean(axis=1)
        n = len(levels) // frame
        cut = lo + int(np.argmin(levels[: n * frame].reshape(n, frame).mean(axis=1))) * frame if n else hi
        parts.append(Region(start, cut))
        start = cut
    parts.append(Region(start, piece.end))
    return parts


def phrase_pieces(audio: np.ndarray, sr: int, segments: list[Region]) -> list[Region]:
    margin = int(PIECE_MARGIN_SECONDS * sr)
    merged = merge_regions(Region(max(0, s.start - margin), min(len(audio), s.end + margin)) for s in segments)
    return [part for piece in merged for part in split_long(audio, sr, piece)]


def apply_pieces(chunk: np.ndarray, sr: int, plan: Optional[AlignPlan], offset: int) -> np.ndarray:
    """Splice the plan's warped pieces into ``chunk`` (absolute start ``offset``)."""
    if plan is None or plan.amount <= 0:
        return chunk
    out = np.array(chunk, dtype=np.float32)
    end = offset + len(chunk)
    fade = max(1, int(PIECE_FADE_SECONDS * sr))
    for piece in plan.pieces:
        lo, hi = max(piece.start, offset), min(piece.end, end)
        if hi <= lo:
            continue
        mask = region_mask(hi - lo, [piece], fade, fades_inside=True, offset=lo)[:, None]
        src = plan.warped(piece)[lo - piece.start : hi - piece.start]
        out[lo - offset : hi - offset] = out[lo - offset : hi - offset] * (1.0 - mask) + src * mask
    return out


def beat_anchors(onsets: np.ndarray, moves: dict[int, float], segments: list[Region], sr: int) -> AnchorCurve:
    """Read-offset anchors in output time: an onset ``o`` moved by ``m`` samples
    (positive = earlier) wants offset ``m`` at ``o - m``; every other onset wants 0.

    Phrase edges (the full piece margin away from the speech) are fixed at 0, so
    the map is the identity wherever pieces meet the untouched audio. The offset
    may change at most MAX_SLOPE per sample in either direction, which keeps
    read positions strictly increasing; targets that cannot be met within that
    slope are reduced, never turned into jumps.
    """
    margin = int(PIECE_MARGIN_SECONDS * sr)
    points: dict[float, tuple[float, bool]] = {}
    for s in segments:
        for edge in (float(max(0, s.start - margin)), float(s.end + margin)):
            points[edge] = (0.0, True)
    for o in onsets:
        move = float(moves.get(int(o), 0.0))
        position = float(o - move)
        if position not in points or not points[position][1]:
            points[position] = (move, False)
    ordered = sorted(points.items())
    positions = np.array([p for p, _ in ordered])
    targets = np.array([t for _, (t, _) in ordered])
    fixed = np.array([f for _, (_, f) in ordered])
    return AnchorCurve(positions, _limit_slope(positions, targets, fixed))


def _limit_slope(positions: np.ndarray, targets: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Closest values to ``targets`` (fixed points held) with |slope| <= MAX_SLOPE."""
    values = np.where(fixed, 0.0, targets)
    gaps = np.diff(positions) * MAX_SLOPE
    for _ in range(len(values) + 1):  # alternate passes until nothing changes
        before = values.copy()
        for i in range(1, len(values)):
            if not fixed[i]:
                values[i] = np.clip(values[i], values[i - 1] - gaps[i - 1], values[i - 1] + gaps[i - 1])
        for i in range(len(values) - 2, -1, -1):
            if not fixed[i]:
                values[i] = np.clip(values[i], values[i + 1] - gaps[i], values[i + 1] + gaps[i])
        if np.array_equal(before, values):
            break
    return values


class BeatAlign(EventStep):
    """Settings: none. The accompaniment comes from the project (B0 / C0)."""

    id = "B7_beats"
    code = "B7"
    title = "讓歌聲跟上拍子"
    description = "以伴奏的拍點為準，把搶拍或慢半拍的字微調回拍子上。音高不變；刻意的唱法可以在清單裡取消。"
    term = "Beat alignment (beat tracking + WSOLA)"
    strength_label = "對齊程度"
    default_strength = 70  # not fully on the grid: some looseness is part of the feel
    options = (Option("sixteenth", "十六分音符（較細）"), Option("eighth", "八分音符"))
    option_label = "對齊的格點"
    default_option = "sixteenth"
    event_label = "搶拍／拖拍"
    uses_accompaniment = True
    shared_keys = ("protected", "accompaniment")
    audition_with_backing = True
    pad_seconds = 0.0
    max_event_seconds = 0.0  # events come from a whole-input measurement
    lead_only = False  # flow C: harmonies follow the lead in the next step instead
    advanced = (ParamSpec("tolerance_ms", "容許範圍（不處理的偏差）", "ms", 5.0, 80.0, 0),)

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {
            "amount": strength / 100.0,
            "tolerance_ms": 30.0,
            "per_beat": 2.0 if (option or self.default_option) == "eighth" else 4.0,
        }

    def measure_fingerprint(self, ctx: StepContext) -> Any:
        return ctx.accompaniment_key

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Optional[dict[str, Any]]:
        backing = ctx.accompaniment()
        if backing is None or (self.lead_only and not ctx.is_lead):
            return None
        segments, _, _ = an.speech_segments(audio, sr)
        return {"grid": beats.track_beats(backing, sr), "onsets": beats.vocal_onsets(audio, sr), "segments": segments}

    def _moves(self, measured: dict[str, Any], params: Params, sr: int) -> list[tuple[int, float]]:
        """(onset, deviation in samples) for every onset outside the tolerance."""
        grid = measured["grid"].subdivide(int(params["per_beat"]))
        onsets = measured["onsets"]
        dev, spacing = beats.deviations(onsets, grid)
        tolerance = params["tolerance_ms"] / 1000.0 * sr
        limit = np.minimum(0.45 * spacing, MAX_MOVE_SECONDS * sr)  # beyond: syncopation, not a mistake
        keep = (np.abs(dev) > tolerance) & (np.abs(dev) <= limit)
        return [(int(o), float(d)) for o, d in zip(onsets[keep], dev[keep])]

    def detect_events(self, audio, sr, params, span=None, measured=None, ctx=None) -> list[Event]:
        if measured is None or not measured["grid"].is_clear:
            return []
        length = int(EVENT_SECONDS * sr)
        events = []
        for onset, deviation in self._moves(measured, params, sr):
            region = Region(onset, onset + length)
            if span is None or region.overlaps(span):
                late = deviation > 0
                events.append(Event(region, f"{'慢' if late else '搶拍'} {abs(deviation) / sr * 1000:.0f} ms"))
        return events

    def detect(self, chunk: np.ndarray, sr: int, params: Params, offset: int) -> list[Region]:
        raise NotImplementedError("B7 detects from the whole-input beat grid (detect_events)")

    def repair(self, chunk: np.ndarray, sr: int, params: Params, events: list[Region], offset: int) -> np.ndarray:
        raise NotImplementedError("B7 splices pre-planned pieces (process)")

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        if self.lead_only and not ctx.is_lead:
            return Analysis(("這是和聲軌：和聲會在下一步跟著（對齊拍點後的）主唱對齊。",), False, self.default_strength)
        measured = self.measure(audio, sr, ctx)
        if measured is None:
            return Analysis(("要先放入伴奏才能對齊拍點（在第一步「放入伴奏」）。沒有伴奏請略過。",), False, self.default_strength)
        grid = measured["grid"]
        if not grid.is_clear:
            return Analysis(
                ("伴奏的拍點不明顯（可能是自由速度的抒情段落），自動對拍容易出錯，建議略過。",), False, self.default_strength
            )
        params = self.resolve_params(None, None, {})
        moves = self._moves(measured, params, sr)
        tolerance = params["tolerance_ms"]
        lines = [f"伴奏速度約 {grid.bpm:.0f} BPM。主唱 {len(measured['onsets'])} 個字中，{len(moves)} 個偏離拍子超過 {tolerance:.0f} ms。"]
        if not moves:
            return Analysis((*lines, "節奏已經很穩，可以略過。"), False, self.default_strength)
        lines.append("清單裡的字可以逐一取消勾選；刻意拖拍或搶拍的唱法請取消。")
        return Analysis(tuple(lines), True, self.default_strength, data={"grid": grid})

    def prepare(self, audio, sr, params, ctx, span, measured) -> Optional[AlignPlan]:
        if measured is None or not measured["grid"].is_clear:
            return None
        chosen = {
            e.region.start for e in self.detect_events(audio, sr, params, None, measured, ctx)
            if not ctx.is_disabled(e.region) and not any(e.region.overlaps(p) for p in ctx.protected)
        }
        moves = {o: d * params["amount"] for o, d in self._moves(measured, params, sr) if o in chosen}
        if not moves:
            return None
        curve = beat_anchors(measured["onsets"], moves, measured["segments"], sr)
        return AlignPlan(audio, sr, curve, 1.0, phrase_pieces(audio, sr, measured["segments"]))

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return apply_pieces(chunk, sr, plan, offset)
