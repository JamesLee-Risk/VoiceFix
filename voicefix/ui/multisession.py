"""Flow C session: one chain per track, the same interface as ``Session``.

The window works on "the current track" (its project and chain); every change
it makes is spread to the other tracks by ``MultiProject.propagate``, and every
render goes through all tracks, lead first.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from voicefix.core.cache import AudioCache, load_accompaniment, load_source
from voicefix.core.chain import Progress, StepChain
from voicefix.core.exporter import ExportRequest, TrackRender, export_multitrack
from voicefix.core.multitrack import LEAD, MultiProject, load_multiproject, save_multiproject
from voicefix.core.project import Flow, Project
from voicefix.core.regions import Region
from voicefix.steps.registry import build_steps, first_affected

log = logging.getLogger(__name__)


class MultiSession:
    is_multitrack = True

    def __init__(self, multi: MultiProject, project_path: Path, cache_root: Path,
                 progress: Optional[Callable[[float], None]] = None) -> None:
        self.project_path = project_path
        self.steps = build_steps(Flow.MULTITRACK)
        ids = [s.id for s in self.steps]
        multi = replace(multi, tracks=tuple(replace(t, project=t.project.migrated(ids)) for t in multi.tracks))
        digest = hashlib.sha1(str(project_path.resolve()).encode("utf-8")).hexdigest()[:16]
        self.chains: list[StepChain] = []
        count = len(multi.tracks)
        for i, track in enumerate(multi.tracks):
            cache = AudioCache(cache_root / digest / f"track{i}")
            report = (lambda f, i=i: progress((i + f) / count)) if progress else None
            source, sr, key = load_source(track.project.source_path, cache, report)
            chain = StepChain(self.steps, cache, source, sr, key, None if i == LEAD else self._reference)
            chain.accompaniment = self._accompaniment
            self.chains.append(chain)
        rates = {c.sr for c in self.chains}
        if len(rates) > 1:
            raise ValueError(f"各軌的取樣率必須相同（目前有 {', '.join(str(r) for r in sorted(rates))} Hz）")
        self.multi = multi
        self._render_lead = threading.local()  # lead snapshot a running render follows
        self.backing: Optional[np.ndarray] = None
        self.backing_key: Optional[str] = None
        self.backing_error: Optional[str] = None
        if multi.accompaniment_path:
            self._open_backing(multi.accompaniment_path)

    @staticmethod
    def create(names: list[str], sources: list[Path], project_path: Path, cache_root: Path,
               progress: Optional[Callable[[float], None]] = None) -> "MultiSession":
        ids = [s.id for s in build_steps(Flow.MULTITRACK)]
        multi = MultiProject.new(names, [str(p.resolve()) for p in sources], ids)
        session = MultiSession(multi, project_path, cache_root, progress)
        session.save()
        return session

    @staticmethod
    def open(project_path: Path, cache_root: Path, progress: Optional[Callable[[float], None]] = None) -> "MultiSession":
        multi = load_multiproject(project_path)
        missing = [t.project.source_path for t in multi.tracks if not Path(t.project.source_path).exists()]
        if missing:
            raise FileNotFoundError(f"找不到錄音：{missing[0]}")
        return MultiSession(multi, project_path, cache_root, progress)

    # ----- the current track ------------------------------------------------
    @property
    def current_track(self) -> int:
        return self.multi.current_track

    @current_track.setter
    def current_track(self, track: int) -> None:
        self.multi = replace(self.multi, current_track=track)
        self.save()

    @property
    def track_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.multi.tracks)

    @property
    def project(self) -> Project:
        return self.multi.project(self.current_track)

    @property
    def chain(self) -> StepChain:
        return self.chains[self.current_track]

    @property
    def cache(self) -> AudioCache:
        return self.chain.cache

    @property
    def sr(self) -> int:
        return self.chains[LEAD].sr

    @property
    def length(self) -> int:
        return max(len(c.source) for c in self.chains)

    @contextmanager
    def following(self, lead_project: Project) -> Iterator[None]:
        """Renders on this thread follow ``lead_project`` (the candidate being applied,
        or a dry lead for dry stems) instead of the committed lead."""
        previous = getattr(self._render_lead, "project", None)
        self._render_lead.project = lead_project
        try:
            yield
        finally:
            self._render_lead.project = previous

    def _reference(self, index: int):
        """The lead's input to step ``index``, for C7 and C12 on the other tracks."""
        lead_project = getattr(self._render_lead, "project", None) or self.multi.project(LEAD)
        lead = self.chains[LEAD]
        key = lead.output_key(lead_project, index - 1) if index > 0 else lead.source_key
        return key, lambda: lead.render(lead_project, index - 1)

    # ----- persistence ------------------------------------------------------
    def update(self, project: Project) -> None:
        self.multi = self.multi.propagate(self.current_track, project)
        self.save()

    def save(self) -> None:
        save_multiproject(self.multi, self.project_path)
        for t, chain in enumerate(self.chains):
            keep = chain.all_keys(self.multi.project(t))
            if t == LEAD and self.backing_key:
                keep.add(self.backing_key)
            chain.cache.prune(keep)

    def go_to(self, index: int) -> None:
        self.update(replace(self.project, current_index=index))

    def set_shared(self, key: str, value: object) -> None:
        self.update(self.project.with_shared(first_affected(self.steps, key), **{key: value}))

    def is_linked(self, step_id: str) -> bool:
        return self.multi.is_linked(self.current_track, step_id)

    def set_linked(self, step_id: str, linked: bool) -> None:
        self.multi = self.multi.with_linked(self.current_track, step_id, linked)
        self.save()

    def set_harmony_gain(self, gain_db: float) -> None:
        self.multi = replace(self.multi, harmony_gain_db=gain_db)
        self.save()

    def to_source(self, index: int, region: Region) -> Region:
        return self.chain.edit_map_before(self.project, index).region_to_source(region)

    # ----- accompaniment ----------------------------------------------------
    def _open_backing(self, path: str) -> None:
        try:
            self.backing, self.backing_key = self.prepare_accompaniment(Path(path))
        except Exception as exc:  # any decode failure: keep working without it
            self.backing, self.backing_key = None, None
            self.backing_error = f"無法讀取伴奏：{exc}" if Path(path).exists() else f"找不到伴奏檔：{path}"
            log.warning(self.backing_error)

    def _accompaniment(self):
        """For steps that follow the beat: (cache key, loader) or None."""
        if self.backing is None:
            return None
        backing = self.backing
        return self.backing_key, lambda: backing

    def prepare_accompaniment(self, path: Path) -> tuple[np.ndarray, str]:
        return load_accompaniment(path, self.chains[LEAD].cache, self.sr, self.length)

    def set_accompaniment(self, path: Optional[Path], prepared: Optional[tuple[np.ndarray, str]] = None) -> None:
        if path is not None:
            self.backing, self.backing_key = prepared or self.prepare_accompaniment(path)
        else:
            self.backing, self.backing_key = None, None
        self.backing_error = None
        resolved = str(path.resolve()) if path else None
        self.multi = replace(self.multi, accompaniment_path=resolved)
        first = first_affected(self.steps, "accompaniment")  # steps that follow the beat must recompute
        self.update(self.project.with_shared(first, accompaniment_path=resolved))

    # ----- rendering --------------------------------------------------------
    def is_ready(self, index: int) -> bool:
        return all(c.is_ready(self.multi.project(t), index) for t, c in enumerate(self.chains))

    def _each_track(self, multi: MultiProject, index: int, progress: Progress, cancel: Optional[threading.Event]) -> None:
        with self.following(multi.project(LEAD)):
            for t, chain in enumerate(self.chains):  # lead first: the others may follow it
                name = multi.tracks[t].name
                chain.render(multi.project(t), index, lambda m, f, name=name: progress(f"{name}：{m}", f), cancel)

    def render_upto(self, index: int, progress: Progress, cancel: Optional[threading.Event]) -> None:
        self._each_track(self.multi, index, progress, cancel)

    def render_step(self, candidate: Project, index: int, progress: Progress, cancel: Optional[threading.Event]) -> None:
        self._each_track(self.multi.propagate(self.current_track, candidate), index, progress, cancel)

    def others_mix(self, index: int, exclude: int) -> Optional[np.ndarray]:
        """Every track but ``exclude``: their input to step ``index``, summed
        (for "其他軌一起播")."""
        if len(self.chains) < 2:
            return None
        total = self.length
        mixed = np.zeros((total, 2), dtype=np.float32)
        for t, chain in enumerate(self.chains):
            if t == exclude:
                continue
            audio = chain.render(self.multi.project(t), index - 1)
            gain = np.float32(10.0 ** (self.multi.harmony_gain_db / 20.0)) if t != LEAD else np.float32(1.0)
            for start in range(0, len(audio), 1 << 20):
                block = np.asarray(audio[start : start + (1 << 20)], dtype=np.float32) * gain
                mixed[start : start + len(block)] += block
        return mixed

    def export(self, project: Project, request: ExportRequest, progress: Progress,
               cancel: Optional[threading.Event]) -> list[Path]:
        multi = self.multi.propagate(self.current_track, project)
        tracks = [
            TrackRender(t.name, self.chains[i], t.project, 0.0 if i == LEAD else multi.harmony_gain_db)
            for i, t in enumerate(multi.tracks)
        ]
        return export_multitrack(
            tracks, request, progress, cancel, self.backing, multi.accompaniment_path, reference_scope=self.following
        )
