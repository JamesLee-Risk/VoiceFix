"""B6 pitch correction and B11 space (reverb).

B6 works on whole notes rather than chunks (see dsp.pitch). Two styles:

* natural: each off-key note gets one constant shift (Rubber Band, formant
  preserving), so vibrato and slides keep their shape.
* effect: every frame of every note is pulled onto the note's target, which
  flattens vibrato (the "auto-tune" sound). This uses TD-PSOLA, because Rubber
  Band's time-varying shift responds late and amplifies vibrato instead.
Settings for B6: {"key": "<key name>" | "chromatic" | "auto"}.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pedalboard

from voicefix.core.regions import Region, region_mask
from voicefix.dsp import pitch
from voicefix.dsp.filters import highpass_sos, zero_phase
from voicefix.dsp.key import estimate_key, note_name, scale_pitch_classes
from voicefix.dsp.psola import psola
from voicefix.steps.base import (
    MAX_NOTE_TARGET_CENTS,
    Analysis,
    Event,
    EventStep,
    Option,
    Params,
    ParamSpec,
    Step,
    StepContext,
)

AUTO_KEY = "auto"
PIECE_FADE_SECONDS = 0.025


@dataclass(frozen=True)
class PitchMeasure:
    track: pitch.PitchTrack
    notes: tuple[pitch.Note, ...]
    estimated_key: str


@dataclass(frozen=True)
class NoteShift:
    region: Region  # the note; its shifted audio replaces exactly this span
    semitones: float  # natural style: one constant shift
    frames: tuple[int, int] = (0, 0)  # effect style: the note's pitch-track frames
    target_cents: float = 0.0  # effect style: where every frame is pulled to


class PitchPlan:
    """Which notes move and by how much. Shifting happens lazily per chunk, so a
    long song never holds all corrected audio in memory; a tiny cache covers
    notes that span two chunks. Shared by preview threads, hence the lock."""

    CACHE_SIZE = 4

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        shifts: list[NoteShift],
        track: Optional[pitch.PitchTrack] = None,
        amount: float = 1.0,
        retune_frames: int = 2,
    ) -> None:
        self.audio, self.sr, self.shifts = audio, sr, shifts
        self.track, self.amount, self.retune_frames = track, amount, retune_frames  # a track means effect style
        self._cache: OrderedDict[Region, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._compute = threading.Lock()  # one shift at a time, so a note is never shifted twice

    def shifted(self, note: NoteShift) -> np.ndarray:
        with self._compute:
            cached = self._cached(note.region)
            return cached if cached is not None else self._shift(note)

    def _cached(self, region: Region) -> Optional[np.ndarray]:
        with self._lock:
            if region in self._cache:
                self._cache.move_to_end(region)
                return self._cache[region]
        return None

    def _shift(self, note: NoteShift) -> np.ndarray:
        outer = pitch.context_region(note.region, self.sr, len(self.audio))
        segment = np.asarray(self.audio[outer.start : outer.end])
        if self.track is None:
            shifted = pitch.shift_segment(segment, self.sr, note.semitones)
        else:
            f0, ratio = pitch.hard_tune(self.track, note.frames, note.target_cents, self.amount, self.retune_frames, outer)
            shifted = psola(segment, self.sr, f0, ratio)
        start = note.region.start - outer.start
        result = shifted[start : start + note.region.length]
        with self._lock:
            self._cache[note.region] = result
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        return result


class PitchCorrection(EventStep):
    id = "B6_pitch"
    code = "B6"
    title = "修正走音"
    description = "找出偏離音階的音，把它們拉回正確的音高。自然模式保留滑音和顫音，只修明顯走音的地方。"
    term = "Pitch correction (Rubber Band, formant preserving)"
    strength_label = "修正程度"
    default_strength = 70
    warn_above = 90
    warn_text = "修正程度太高會讓歌聲聽起來不自然。自然模式建議 50～80。"
    options = (Option("natural", "自然（只修明顯走音）"), Option("effect", "明顯效果（電音感，拉平顫音）"))
    option_label = "風格"
    default_option = "natural"
    scopes = ()  # applied per detected note
    panel = "pitch"
    event_label = "走音"
    pad_seconds = 0.1
    max_event_seconds = 0.0  # notes come from the whole-input track, so a span finds every overlapping note
    advanced = (
        ParamSpec("threshold_cents", "偏離多少才修正", "音分", 0.0, 100.0, 0),
        ParamSpec("retune_ms", "拉回速度（明顯效果）", "ms", 10.0, 200.0, 0),
    )

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        effect = (option or self.default_option) == "effect"
        return {
            "amount": strength / 100.0,
            # PRD: natural mode fixes notes >= 30 cents off (initial value, to be tuned);
            # the effect style processes every note
            "threshold_cents": 0.0 if effect else 30.0,
            "retune_ms": 30.0,
            "effect": 1.0 if effect else 0.0,
        }

    # ----- measurement and events -----------------------------------------
    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> PitchMeasure:
        track = pitch.track_pitch(audio, sr)
        key, _ = estimate_key(audio, sr)
        return PitchMeasure(track, tuple(pitch.segment_notes(track)), key)

    @staticmethod
    def key_for(ctx: Optional[StepContext], measured: PitchMeasure) -> str:
        chosen = (ctx.settings.get("key") if ctx else None) or AUTO_KEY
        return measured.estimated_key if chosen == AUTO_KEY else chosen

    def _deviations(
        self, measured: PitchMeasure, key: str, ctx: Optional[StepContext] = None
    ) -> list[tuple[pitch.Note, float, float, bool]]:
        """(note, cents off its target, target, chosen by hand) for every note.

        Without a hand-picked target the nearest in-key note is assumed, which is
        wrong once a note is more than half a semitone off: then the neighbour
        is nearer. That is what the per-note targets are for."""
        classes = scale_pitch_classes(key)
        out = []
        for note in measured.notes:
            manual = ctx.target_for(note.region) if ctx else None
            if manual is not None and abs(manual - note.cents) > MAX_NOTE_TARGET_CENTS:
                manual = None  # left over from an edit that moved the note: never shift that far
            target = float(manual) if manual is not None else float(pitch.nearest_in_scale(note.cents, classes))
            out.append((note, note.cents - target, target, manual is not None))
        return out

    def detect_events(self, audio, sr, params, span=None, measured=None, ctx=None) -> list[Event]:
        measured = measured or self.measure(audio, sr, ctx)
        events = []
        for note, deviation, target, manual in self._deviations(measured, self.key_for(ctx, measured), ctx):
            if (abs(deviation) < params["threshold_cents"] and not manual) or (span and not note.region.overlaps(span)):
                continue
            direction = "偏高" if deviation > 0 else "偏低"
            source = "，手動" if manual else ""
            events.append(Event(note.region, f"{direction} {abs(deviation):.0f} 音分（{note_name(target)}{source}）"))
        return events

    def detect(self, chunk: np.ndarray, sr: int, params: Params, offset: int) -> list[Region]:
        raise NotImplementedError("B6 detects from the whole-input pitch track (detect_events)")

    def repair(self, chunk: np.ndarray, sr: int, params: Params, events: list[Region], offset: int) -> np.ndarray:
        raise NotImplementedError("B6 crossfades pre-rendered pieces (process)")

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        measured = self.measure(audio, sr, ctx)
        key = self.key_for(ctx, measured)
        params = self.resolve_params(None, None, {})
        off = self.detect_events(audio, sr, params, measured=measured, ctx=ctx)
        source = "自動偵測" if (ctx.settings.get("key") or AUTO_KEY) == AUTO_KEY else "你選的"
        lines = [f"調性：{key}（{source}）。共 {len(measured.notes)} 個音，其中 {len(off)} 個偏離 30 音分以上。"]
        classes = scale_pitch_classes(key)
        notes = [
            {"region": n.region, "cents": n.cents, "target": t, "manual": m,
             "auto_target": float(pitch.nearest_in_scale(n.cents, classes))}
            for n, _, t, m in self._deviations(measured, key, ctx)
        ]
        data = {"pitch": measured.track, "key": key, "estimated_key": measured.estimated_key, "notes": notes}
        if not off:
            return Analysis((*lines, "音準已經不錯，可以略過。"), False, self.default_strength, data=data)
        lines.append("清單裡的音可以逐一取消勾選；刻意的滑音或藍調音請取消。")
        lines.append("曲線上每個音標著程式認定的目標音；認錯的音（偏太多時會被當成隔壁的音）請點選後用 ↑↓ 改正。")
        return Analysis(tuple(lines), True, self.default_strength, data=data)

    # ----- rendering --------------------------------------------------------
    def prepare(self, audio, sr, params, ctx, span, measured: PitchMeasure) -> PitchPlan:
        events = self.detect_events(audio, sr, params, span, measured, ctx)
        chosen = {
            e.region for e in events
            if not ctx.is_disabled(e.region) and not any(e.region.overlaps(p) for p in ctx.protected)
        }
        shifts = [
            NoteShift(n.region, -d / 100.0 * params["amount"], n.frames, target)
            for n, d, target, _ in self._deviations(measured, self.key_for(ctx, measured), ctx)
            if n.region in chosen
        ]
        if not params["effect"]:
            return PitchPlan(audio, sr, shifts)
        retune = max(1, int(params["retune_ms"] / 1000.0 / pitch.FRAME_SECONDS))
        return PitchPlan(audio, sr, shifts, measured.track, params["amount"], retune)

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        out = np.array(chunk, dtype=np.float32)
        end = offset + len(chunk)
        fade = max(1, int(PIECE_FADE_SECONDS * sr))
        for note in plan.shifts:
            lo, hi = max(note.region.start, offset), min(note.region.end, end)
            if hi <= lo:
                continue
            mask = region_mask(hi - lo, [note.region], fade, fades_inside=True, offset=lo)[:, None]
            src = plan.shifted(note)[lo - note.region.start : hi - note.region.start]
            out[lo - offset : hi - offset] = out[lo - offset : hi - offset] * (1.0 - mask) + src * mask
        return out


SPACE_PRESETS = {
    "small_room": {"room_size": 0.35, "damping": 0.6, "predelay": 0.012, "width": 0.8},
    "hall": {"room_size": 0.75, "damping": 0.4, "predelay": 0.035, "width": 1.0},
}
WET_HIGHPASS_HZ = 180.0  # keep the low end of the reverb from muddying the vocal


class Space(Step):
    id = "B11_space"
    code = "B11"
    title = "加一點空間"
    description = "為乾乾的歌聲加一點房間或大廳的空間感。乾聲（不含空間效果）永遠可以另外匯出給混音師。"
    term = "Reverb"
    strength_label = "量"
    default_strength = 25
    warn_above = 70
    warn_text = "空間感太多會讓歌聲變遠、字聽不清楚。"
    options = (Option("small_room", "小房間"), Option("hall", "大廳"))
    option_label = "空間"
    default_option = "small_room"
    scopes = ()
    respects_protection = False
    shared_keys = ()
    audition_with_backing = True
    wet_effect = True
    pad_seconds = 6.0  # longer than the hall's decay, so chunks match the whole file
    advanced = (ParamSpec("wet_db", "殘響音量", "dB", -40.0, 0.0, 1),)

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {
            "wet_db": -30.0 + 0.3 * strength,  # 25 -> -22.5 dB: a touch, not a wash
            "hall": 1.0 if (option or self.default_option) == "hall" else 0.0,
        }

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        return Analysis(
            ("空間感看個人喜好。如果要交給混音師，通常可以略過，讓混音師自己加。",),
            False,
            self.default_strength,
            self.default_option,
        )

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        preset = SPACE_PRESETS["hall" if params["hall"] >= 0.5 else "small_room"]
        reverb = pedalboard.Reverb(
            room_size=preset["room_size"], damping=preset["damping"], wet_level=1.0, dry_level=0.0, width=preset["width"]
        )
        dry = np.asarray(chunk, dtype=np.float32)
        wet = reverb(np.ascontiguousarray(dry.T), sr).T
        delay = int(preset["predelay"] * sr)
        wet = np.concatenate([np.zeros((delay, dry.shape[1]), np.float32), wet[: len(wet) - delay]])
        wet = zero_phase(highpass_sos(sr, WET_HIGHPASS_HZ, 2), wet)
        return (dry + wet * np.float32(10.0 ** (params["wet_db"] / 20.0))).astype(np.float32)
