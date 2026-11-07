"""The step chain: renders step outputs, decides what needs recomputing.

Rendering is a pure function of (source, step configs), which is what makes a
reopened project export bit-identical results.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from voicefix.core.cache import AudioCache, make_key
from voicefix.core.chunking import CHUNK_SECONDS, check_cancel, iter_chunks
from voicefix.core.editmap import EditMap, cut_block
from voicefix.core.project import Project, StepConfig
from voicefix.core.regions import Region, ScopeMode, blend, fade_samples, keep_mask
from voicefix.steps.base import Analysis, Step, StepContext

Progress = Callable[[str, float], None]
# flow C: step index -> (cache key, loader) of the lead track's input to that step
ReferenceProvider = Callable[[int], tuple[str, Callable[[], np.ndarray]]]
# the accompaniment: (cache key, loader), or None when the project has none
AccompanimentProvider = Callable[[], Optional[tuple[str, Callable[[], np.ndarray]]]]
MEMO_SIZE = 16
CUT_FADE_SECONDS = 0.005


@dataclass(frozen=True)
class Preview:
    before: np.ndarray
    after: np.ndarray
    span: Region


def noop_progress(message: str, fraction: float) -> None:
    pass


class _Memo:
    def __init__(self, size: int) -> None:
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = compute()
        with self._lock:
            self._items[key] = value
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return value


class StepChain:
    def __init__(
        self,
        steps: Sequence[Step],
        cache: AudioCache,
        source: np.ndarray,
        sr: int,
        source_key: str,
        reference: Optional[ReferenceProvider] = None,
    ) -> None:
        self.steps = list(steps)
        self.reference = reference  # set on flow C harmony tracks; None on the lead
        self.accompaniment: Optional[AccompanimentProvider] = None  # set by the session
        self.cache = cache
        self.source = source
        self.sr = sr
        self.source_key = source_key
        self._memo = _Memo(MEMO_SIZE)
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._key_locks_guard:
            return self._key_locks.setdefault(key, threading.Lock())

    # ----- timeline ---------------------------------------------------------
    def edit_map_before(self, project: Project, index: int) -> EditMap:
        edit_map = EditMap()
        for i in range(min(index, len(self.steps))):
            cfg = project.config(i)
            if self.steps[i].changes_length and cfg.status.is_active:
                edit_map = self.steps[i].edit_map(cfg.settings)
        return edit_map

    def input_length(self, project: Project, index: int) -> int:
        return self.edit_map_before(project, index).edited_length(len(self.source))

    def context(self, project: Project, index: int, config: Optional[StepConfig] = None) -> StepContext:
        config = config or project.config(index)
        edit_map = self.edit_map_before(project, index)
        noise = edit_map.region_to_edited(project.noise_sample) if project.noise_sample else None
        return StepContext(
            sr=self.sr,
            total=edit_map.edited_length(len(self.source)),
            selections=tuple(edit_map.regions_to_edited(config.selections)),
            protected=tuple(edit_map.regions_to_edited(project.protected)),
            noise_sample=noise,
            binaural=project.binaural,
            disabled_events=tuple(edit_map.to_edited(p) for p in config.disabled_events),
            note_targets=tuple((edit_map.to_edited(p), c) for p, c in config.note_targets),
            settings=config.settings,
            edit_map=edit_map,
            **self._reference_fields(index),
        )

    def _reference_fields(self, index: int) -> dict[str, Any]:
        fields: dict[str, Any] = {"is_lead": self.reference is None}
        step = self.steps[index]
        if self.reference is not None and step.uses_reference:
            fields["reference_key"], fields["reference_loader"] = self.reference(index)
        if step.uses_accompaniment and self.accompaniment is not None:
            provided = self.accompaniment()
            if provided is not None:
                fields["accompaniment_key"], fields["accompaniment_loader"] = provided
        return fields

    def _accompaniment_key(self) -> Optional[str]:
        provided = self.accompaniment() if self.accompaniment is not None else None
        return provided[0] if provided else None

    # ----- cache keys -------------------------------------------------------
    def _step_key(self, prev: str, project: Project, index: int, config: StepConfig, force: bool) -> str:
        step = self.steps[index]
        if step.passthrough or not (force or config.status.is_active):
            return prev
        parts = [prev, step.id, step.version, config.fingerprint(), project.shared_fingerprint(step.shared_keys)]
        if self.reference is not None and step.uses_reference:
            parts.append(self.reference(index)[0])  # the lead's input: follow it when it changes
        if step.uses_accompaniment:
            parts.append(self._accompaniment_key())  # the beat grid comes from it
        return make_key(*parts)

    def output_key(self, project: Project, index: int) -> str:
        key = self.source_key
        for i in range(index + 1):
            key = self._step_key(key, project, i, project.config(i), force=False)
        return key

    def all_keys(self, project: Project) -> set[str]:
        return {self.source_key} | {self.output_key(project, i) for i in range(len(self.steps))}

    def is_ready(self, project: Project, index: int) -> bool:
        """Whether the output of step ``index`` (-1 = source) is already cached."""
        key = self.output_key(project, index) if index >= 0 else self.source_key
        return key == self.source_key or self.cache.has(key)

    # ----- rendering --------------------------------------------------------
    def render(
        self,
        project: Project,
        index: int,
        progress: Progress = noop_progress,
        cancel: Optional[threading.Event] = None,
    ) -> np.ndarray:
        """Output of step ``index`` (-1 = source), computing missing steps in order."""
        audio, key = self.source, self.source_key
        for i in range(index + 1):
            next_key = self._step_key(key, project, i, project.config(i), force=False)
            if next_key != key:
                with self._lock_for(next_key):  # never compute the same output twice at once
                    cached = self.cache.get(next_key)
                    if cached is None:
                        cached = self._run_step(project, i, project.config(i), audio, next_key, progress, cancel)
                audio = cached
            key = next_key
        return audio

    def _run_step(
        self,
        project: Project,
        index: int,
        config: StepConfig,
        audio: np.ndarray,
        key: str,
        progress: Progress,
        cancel: Optional[threading.Event],
    ) -> np.ndarray:
        step = self.steps[index]
        label = f"{step.code} {step.title}"
        progress(f"{label}：準備中", 0.0)
        ctx = self.context(project, index, config)
        params = step.resolve_params(config.strength, config.option, config.advanced)
        measured = self._measure(project, index, audio, ctx)
        plan = step.prepare(audio, self.sr, params, ctx, None, measured)
        edit_map = step.edit_map(config.settings) if step.changes_length else EditMap()
        total = len(audio)
        cut_fade = fade_samples(self.sr, CUT_FADE_SECONDS)
        chunks = list(iter_chunks(total, int(CHUNK_SECONDS * self.sr), int(step.pad_seconds * self.sr)))
        with self.cache.writer(key, (edit_map.edited_length(total), audio.shape[1])) as out:
            for n, chunk in enumerate(chunks):
                check_cancel(cancel)
                block = self._process_span(step, audio, params, ctx, plan, config, chunk.core)
                if edit_map.is_identity:
                    out.write(block)
                else:
                    for piece in cut_block(edit_map, block, chunk.core.start, total, cut_fade):
                        out.write(piece)
                progress(f"{label}：處理中", (n + 1) / len(chunks))
        result = self.cache.get(key)
        if result is None:
            raise RuntimeError(f"{label}：無法寫入暫存檔（{self.cache.directory}）")
        return result

    def _process_span(
        self,
        step: Step,
        audio: np.ndarray,
        params: dict[str, float],
        ctx: StepContext,
        plan: Any,
        config: StepConfig,
        core: Region,
    ) -> np.ndarray:
        pad = int(step.pad_seconds * self.sr)
        lo, hi = max(0, core.start - pad), min(len(audio), core.end + pad)
        padded = np.asarray(audio[lo:hi], dtype=np.float32)
        processed = step.process(padded, self.sr, params, ctx, plan, lo)
        original = padded[core.start - lo : core.end - lo]
        result = processed[core.start - lo : core.end - lo]
        protected = ctx.protected if step.respects_protection else ()
        scope = config.scope if config.scope in step.scopes else ScopeMode.WHOLE
        keep = keep_mask(core.length, core.start, scope, ctx.selections, protected, fade_samples(self.sr))
        return blend(original, result, keep)

    # ----- interactive ------------------------------------------------------
    def _input_key(self, project: Project, index: int) -> str:
        return self.output_key(project, index - 1) if index > 0 else self.source_key

    @staticmethod
    def _ctx_fingerprint(ctx: StepContext) -> str:
        return json.dumps(
            {
                "sel": [r.to_json() for r in ctx.selections],
                "prot": [r.to_json() for r in ctx.protected],
                "noise": ctx.noise_sample.to_json() if ctx.noise_sample else None,
                "dis": list(ctx.disabled_events),
                "targets": [list(t) for t in ctx.note_targets],
                "settings": ctx.settings,
                "ref": ctx.reference_key,
                "acc": ctx.accompaniment_key,
                "lead": ctx.is_lead,
            },
            sort_keys=True,
            default=str,
        )

    def _measure(self, project: Project, index: int, audio: np.ndarray, ctx: StepContext) -> Any:
        step = self.steps[index]
        memo_key = make_key("measure", self._input_key(project, index), step.id, step.version, step.measure_fingerprint(ctx))
        return self._memo.get_or_compute(memo_key, lambda: step.measure(audio, self.sr, ctx))

    def analyze(
        self,
        project: Project,
        index: int,
        cancel: Optional[threading.Event] = None,
        config: Optional[StepConfig] = None,
    ) -> Analysis:
        """``config`` lets the screen analyse with draft step settings (e.g. B6's key)."""
        step = self.steps[index]
        audio = self.render(project, index - 1, cancel=cancel)
        ctx = self.context(project, index, config)
        memo_key = make_key("analysis", self._input_key(project, index), step.id, step.version, self._ctx_fingerprint(ctx))
        return self._memo.get_or_compute(memo_key, lambda: step.analyze(audio, self.sr, ctx))

    def preview(
        self,
        project: Project,
        index: int,
        config: StepConfig,
        span: Region,
        cancel: Optional[threading.Event] = None,
    ) -> Preview:
        """Process only ``span`` (plus padding) of the step input with a draft config."""
        step = self.steps[index]
        audio = self.render(project, index - 1, cancel=cancel)
        span = Region(max(0, span.start), min(len(audio), span.end))
        before = np.array(audio[span.start : span.end], dtype=np.float32)
        if step.passthrough or span.length == 0:
            return Preview(before, before, span)
        ctx = self.context(project, index, config)
        params = step.resolve_params(config.strength, config.option, config.advanced)
        plan_span = None
        if step.event_based:  # include events that start before the span but reach into it
            plan_span = Region(max(0, span.start - int(step.max_event_seconds * self.sr)), span.end)  # 0 = exact span
        memo_key = make_key(
            "plan", self._input_key(project, index), step.id, step.version, params, self._ctx_fingerprint(ctx),
            plan_span.to_json() if plan_span else None,
        )
        measured = self._measure(project, index, audio, ctx)
        plan = self._memo.get_or_compute(memo_key, lambda: step.prepare(audio, self.sr, params, ctx, plan_span, measured))
        check_cancel(cancel)
        after = self._process_span(step, audio, params, ctx, plan, config, span)
        return Preview(before, after, span)

    def events(self, project: Project, index: int, config: StepConfig, cancel: Optional[threading.Event] = None) -> list:
        step = self.steps[index]
        if not step.event_based:
            return []
        audio = self.render(project, index - 1, cancel=cancel)
        params = step.resolve_params(config.strength, config.option, config.advanced)
        ctx = self.context(project, index, config)
        measured = self._measure(project, index, audio, ctx)
        memo_key = make_key(
            "events", self._input_key(project, index), step.id, step.version, params, self._ctx_fingerprint(ctx)
        )
        return self._memo.get_or_compute(
            memo_key, lambda: step.detect_events(audio, self.sr, params, measured=measured, ctx=ctx)
        )
