"""The Step contract every processing step implements.

The chain owns scope blending, protected regions, chunking and caching; a step
only defines analysis, the strength-to-parameter mapping and the DSP itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

from voicefix.core.chunking import detect_chunked
from voicefix.core.editmap import EditMap
from voicefix.core.regions import Region, ScopeMode, clip_regions

Params = Mapping[str, float]

GENERAL_SCOPES = (ScopeMode.WHOLE, ScopeMode.SELECTION, ScopeMode.EXCLUDE_SELECTION)
EVENT_MATCH_SECONDS = 0.002
NOTE_MATCH_SECONDS = 0.03  # a note's start may move this much when earlier steps change
MAX_NOTE_TARGET_CENTS = 350  # a hand-picked target may be at most ~3 semitones from what was sung


@dataclass(frozen=True)
class Option:
    key: str
    label: str


@dataclass(frozen=True)
class ParamSpec:
    """An advanced parameter shown with its real unit (Hz, dB, ...)."""

    key: str
    label: str
    unit: str
    minimum: float
    maximum: float
    decimals: int = 1


@dataclass(frozen=True)
class Event:
    region: Region  # step-input timeline
    label: str


@dataclass(frozen=True)
class Analysis:
    summary: tuple[str, ...]
    recommend: bool
    suggested_strength: Optional[int] = None
    suggested_option: Optional[str] = None
    marks: tuple[Region, ...] = ()  # highlighted on the waveform (step-input timeline)
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepContext:
    """Project state mapped into this step's input timeline."""

    sr: int
    total: int
    selections: tuple[Region, ...]
    protected: tuple[Region, ...]
    noise_sample: Optional[Region]
    binaural: bool
    disabled_events: tuple[int, ...]
    settings: Mapping[str, Any]
    edit_map: EditMap = EditMap()
    # flow C: the lead track's input to this step, for steps that follow the lead
    reference_key: Optional[str] = None
    reference_loader: Optional[Callable[[], np.ndarray]] = None
    # the accompaniment (flows B and C), for steps that follow its beat
    accompaniment_key: Optional[str] = None
    accompaniment_loader: Optional[Callable[[], np.ndarray]] = None
    is_lead: bool = True  # flow C: False on harmony / double tracks
    note_targets: tuple[tuple[int, int], ...] = ()  # (note start, target cents) chosen by hand

    def reference(self) -> Optional[np.ndarray]:
        return self.reference_loader() if self.reference_loader else None

    def accompaniment(self) -> Optional[np.ndarray]:
        return self.accompaniment_loader() if self.accompaniment_loader else None

    def target_for(self, note: Region) -> Optional[int]:
        """The hand-picked target (cents) of the note starting at ``note.start``, if any."""
        tolerance = int(self.sr * NOTE_MATCH_SECONDS)
        nearest = min(self.note_targets, key=lambda t: abs(note.start - t[0]), default=None)
        if nearest is None or abs(note.start - nearest[0]) > tolerance:
            return None
        return nearest[1]

    def is_disabled(self, event: Region) -> bool:
        tolerance = int(self.sr * EVENT_MATCH_SECONDS)
        return any(abs(event.start - pos) <= tolerance for pos in self.disabled_events)


class Step(ABC):
    id: str = ""
    code: str = ""
    title: str = ""  # plain-language name shown on screen
    description: str = ""  # one sentence: what this step solves
    term: str = ""  # jargon, shown in small print only
    version: int = 1  # bump when the DSP changes, to invalidate caches

    has_strength: bool = True
    strength_label: str = "強度"
    default_strength: int = 50
    warn_above: Optional[int] = None
    warn_text: str = ""

    options: tuple[Option, ...] = ()
    option_label: str = ""
    default_option: Optional[str] = None

    scopes: tuple[ScopeMode, ...] = GENERAL_SCOPES
    respects_protection: bool = True
    shared_keys: tuple[str, ...] = ("protected",)
    event_based: bool = False
    passthrough: bool = False  # analysis-only steps: output == input
    changes_length: bool = False
    pad_seconds: float = 2.0
    advanced: tuple[ParamSpec, ...] = ()

    panel: Optional[str] = None  # extra UI: "import", "edit" or "export"
    audition_with_backing: bool = False  # flow B tone steps: listen inside the accompaniment
    wet_effect: bool = False  # spatial effects, left out of the dry export
    uses_reference: bool = False  # flow C: output depends on the lead track (C7b, C12)
    uses_accompaniment: bool = False  # output depends on the accompaniment (B7, C7a)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        return Analysis(summary=(), recommend=True, suggested_strength=self.default_strength)

    def map_strength(self, strength: int, option: Optional[str]) -> dict[str, float]:
        return {}

    def resolve_params(self, strength: Optional[int], option: Optional[str], advanced: Params) -> dict[str, float]:
        """Strength mapping with advanced overrides on top."""
        params = self.map_strength(self.default_strength if strength is None else strength, option or self.default_option)
        params.update({k: v for k, v in advanced.items() if k in params})
        return params

    def measure(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Any:
        """Parameter-independent whole-input measurement; memoized by the chain."""
        return None

    def measure_fingerprint(self, ctx: StepContext) -> Any:
        """The parts of ``ctx`` that measure() reads (memo key). Most steps read none,
        so ticking an event or drawing a selection never repeats a long measurement."""
        return None

    def prepare(
        self, audio: np.ndarray, sr: int, params: Params, ctx: StepContext, span: Optional[Region], measured: Any
    ) -> Any:
        """Per-parameter work (profiles, event lists, gains) before chunked processing.
        ``span`` limits the work to a preview range when the step can do so."""
        return measured

    @abstractmethod
    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        """Process one chunk whose first sample is at ``offset`` in the input."""

    def edit_map(self, settings: Mapping[str, Any]) -> EditMap:
        return EditMap()

    def warning(self, strength: int, option: Optional[str] = None) -> Optional[str]:
        """A concrete consequence to show when the settings leave the safe range."""
        if self.warn_above is not None and strength > self.warn_above:
            return self.warn_text
        return None


class PassthroughStep(Step):
    passthrough = True
    has_strength = False
    shared_keys = ()
    respects_protection = False
    scopes = ()

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        return chunk


class EventStep(Step):
    """Detect short events, let the user untick some, repair the rest."""

    event_based = True
    scopes = (ScopeMode.EVENTS,)
    event_label: str = "事件"
    detect_pad_seconds: float = 0.5  # context a detector needs around each chunk
    max_event_seconds: float = 0.25  # longest event; previews look this far back

    @abstractmethod
    def detect(self, chunk: np.ndarray, sr: int, params: Params, offset: int) -> list[Region]: ...

    @abstractmethod
    def repair(self, chunk: np.ndarray, sr: int, params: Params, events: list[Region], offset: int) -> np.ndarray: ...

    def kind_label(self, region: Region, sr: int) -> str:
        """Name shown for one event; combined steps override this."""
        return self.event_label

    def detect_events(
        self,
        audio: np.ndarray,
        sr: int,
        params: Params,
        span: Optional[Region] = None,
        measured: Any = None,
        ctx: Optional[StepContext] = None,
    ) -> list[Event]:
        """Events in ``span`` (or the whole input). ``measured`` and ``ctx`` are for
        detectors that work from a whole-input measurement instead of chunks."""
        regions = detect_chunked(
            audio, sr, lambda c, off: self.detect(c, sr, params, off), span=span, pad_seconds=self.detect_pad_seconds
        )
        return [Event(r, f"{self.kind_label(r, sr)} {i + 1}") for i, r in enumerate(regions)]

    def prepare(
        self, audio: np.ndarray, sr: int, params: Params, ctx: StepContext, span: Optional[Region], measured: Any
    ) -> Any:
        events = [e.region for e in self.detect_events(audio, sr, params, span, measured, ctx)]
        return [
            r
            for r in events
            if not ctx.is_disabled(r) and not (self.respects_protection and any(r.overlaps(p) for p in ctx.protected))
        ]

    def process(self, chunk: np.ndarray, sr: int, params: Params, ctx: StepContext, plan: Any, offset: int) -> np.ndarray:
        events = clip_regions(plan, offset, offset + len(chunk))
        return self.repair(chunk, sr, params, events, offset)

    def analyze(self, audio: np.ndarray, sr: int, ctx: StepContext) -> Analysis:
        params = self.resolve_params(self.default_strength, self.default_option, {})
        events = self.detect_events(audio, sr, params)
        minutes = max(len(audio) / sr / 60.0, 1e-9)
        if not events:
            return Analysis((f"沒有偵測到{self.event_label}。",), recommend=False, suggested_strength=self.default_strength)
        return Analysis(
            (f"以預設敏感度偵測到 {len(events)} 個{self.event_label}（每分鐘約 {len(events) / minutes:.1f} 個）。",),
            recommend=True,
            suggested_strength=self.default_strength,
        )
