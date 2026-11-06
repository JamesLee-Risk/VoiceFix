"""Render the whole chain and write the result(s) to disk."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from contextlib import nullcontext
from typing import Callable, ContextManager, Iterator, Optional, Sequence

import numpy as np

from voicefix.core import analysis as an
from voicefix.core import audio_io
from voicefix.core.chain import Progress, StepChain, noop_progress
from voicefix.core.chunking import CHUNK_SECONDS, check_cancel, iter_chunks
from voicefix.core.project import Project, StepStatus
from voicefix.dsp.dynamics import limit

MIX_CEILING_DBTP = -1.0
MIX_PAD_SECONDS = 0.2
DRY_SUFFIX = "_dry"
MIX_SUFFIX = "_mix"


@dataclass(frozen=True)
class ExportRequest:
    path: Path  # target file; with ``split`` it is used as the name stem
    format_key: str = "wav24"
    split: bool = False
    dry: bool = False  # also export without spatial effects (flow B)
    mix: bool = False  # also export a preview mixed with the accompaniment (flow B)


def without_wet_effects(chain: StepChain, project: Project) -> Project:
    """The same project with every spatial-effect step treated as skipped."""
    for i, step in enumerate(chain.steps):
        if step.wet_effect and project.status(i).is_active:
            project = project.with_config(i, replace(project.config(i), status=StepStatus.SKIPPED))
    return project


def has_wet_effects(chain: StepChain, project: Project) -> bool:
    return any(step.wet_effect and project.status(i).is_active for i, step in enumerate(chain.steps))


def mixed_blocks(vocal: np.ndarray, backing: np.ndarray, sr: int) -> Iterator[np.ndarray]:
    """Vocal + accompaniment, limited to MIX_CEILING_DBTP, in chunks."""
    pad = int(MIX_PAD_SECONDS * sr)
    for chunk in iter_chunks(len(vocal), int(CHUNK_SECONDS * sr), pad):
        lo, hi = chunk.padded.start, chunk.padded.end
        mixed = np.asarray(vocal[lo:hi], dtype=np.float32) + np.asarray(backing[lo:hi], dtype=np.float32)
        limited = limit(mixed, sr, MIX_CEILING_DBTP)
        yield limited[chunk.core.start - lo : chunk.core.end - lo]


def export(
    chain: StepChain,
    project: Project,
    request: ExportRequest,
    progress: Progress = noop_progress,
    cancel: Optional[threading.Event] = None,
    backing: Optional[np.ndarray] = None,
) -> list[Path]:
    """Render every active step (recomputing stale ones) and write the file(s)."""
    fmt = audio_io.export_format(request.format_key)
    target = request.path.with_suffix(fmt.suffix)
    guard = _OverwriteGuard(project)
    guard.check(target)

    audio = chain.render(project, len(chain.steps) - 1, progress, cancel)
    check_cancel(cancel)
    written = _write_main(chain, audio, target, fmt, request.split, progress, cancel, guard)
    if request.dry and has_wet_effects(chain, project):
        dry = chain.render(without_wet_effects(chain, project), len(chain.steps) - 1, progress, cancel)
        path = guard.check(target.with_name(f"{target.stem}{DRY_SUFFIX}{fmt.suffix}"))
        audio_io.write_audio(path, dry, chain.sr, fmt, lambda f: progress("寫入乾聲", f))
        written.append(path)
    if request.mix:
        if backing is None:
            raise ValueError("沒有伴奏，無法匯出混合試聽版。請先在 B0 放入伴奏。")
        path = guard.check(target.with_name(f"{target.stem}{MIX_SUFFIX}{fmt.suffix}"))
        channels = max(audio.shape[1], backing.shape[1])
        blocks = mixed_blocks(audio, backing, chain.sr)  # a mono vocal broadcasts per block
        audio_io.write_blocks(path, blocks, len(audio), channels, chain.sr, fmt, lambda f: progress("寫入混合試聽版", f))
        written.append(path)
    return written


class _OverwriteGuard:
    """No export may replace the source recording or the accompaniment."""

    def __init__(self, project: Project, extra: Sequence[Optional[str]] = ()) -> None:
        inputs = [project.source_path, project.accompaniment_path, *extra]
        self._inputs = {Path(p).resolve() for p in inputs if p}

    def check(self, path: Path) -> Path:
        if path.resolve() in self._inputs:
            raise ValueError(f"不能覆寫原始錄音或伴奏（{path.name}），請換一個檔名。")
        return path


def _write_main(
    chain: StepChain,
    audio: np.ndarray,
    target: Path,
    fmt: audio_io.ExportFormat,
    split: bool,
    progress: Progress,
    cancel: Optional[threading.Event],
    guard: _OverwriteGuard,
) -> list[Path]:
    if not split:
        audio_io.write_audio(target, audio, chain.sr, fmt, lambda f: progress("寫入檔案", f))
        return [target]
    segments, _, _ = an.speech_segments(audio, chain.sr)
    written = []
    for n, segment in enumerate(segments, start=1):
        check_cancel(cancel)
        path = guard.check(target.with_name(f"{target.stem}_{n:03d}{fmt.suffix}"))
        audio_io.write_audio(path, audio[segment.start : segment.end], chain.sr, fmt)
        written.append(path)
        progress("分段寫入", n / len(segments))
    return written


@dataclass(frozen=True)
class TrackRender:
    name: str
    chain: StepChain
    project: Project
    gain_db: float  # level in the mix preview


def mixed_tracks_blocks(tracks: list[tuple[np.ndarray, float]], backing: Optional[np.ndarray], sr: int, total: int) -> Iterator[np.ndarray]:
    """Sum of (audio, linear gain) tracks plus accompaniment, limited, in chunks."""
    pad = int(MIX_PAD_SECONDS * sr)
    for chunk in iter_chunks(total, int(CHUNK_SECONDS * sr), pad):
        lo, hi = chunk.padded.start, chunk.padded.end
        mixed = np.zeros((hi - lo, 2), dtype=np.float32)
        for audio, gain in tracks:
            part = np.asarray(audio[lo:hi], dtype=np.float32) * np.float32(gain)
            mixed[: len(part)] += part  # a mono track broadcasts to both sides
        if backing is not None:
            part = np.asarray(backing[lo:hi], dtype=np.float32)
            mixed[: len(part)] += part
        yield limit(mixed, sr, MIX_CEILING_DBTP)[chunk.core.start - lo : chunk.core.end - lo]


def export_multitrack(
    tracks: list[TrackRender],
    request: ExportRequest,
    progress: Progress = noop_progress,
    cancel: Optional[threading.Event] = None,
    backing: Optional[np.ndarray] = None,
    accompaniment_path: Optional[str] = None,
    reference_scope: Callable[[Project], ContextManager[None]] = lambda lead: nullcontext(),
) -> list[Path]:
    """Flow C: one cleaned file per track, optional dry stems and a mix preview.
    ``reference_scope(lead)`` makes renders follow that lead (C7 / C12), so dry
    stems use the dry lead's loudness like the lead's own dry stem does."""
    fmt = audio_io.export_format(request.format_key)
    target = request.path.with_suffix(fmt.suffix)
    guard = _OverwriteGuard(tracks[0].project, [t.project.source_path for t in tracks] + [accompaniment_path])
    written, rendered = [], []
    lead = tracks[0]
    dry_lead = without_wet_effects(lead.chain, lead.project)
    for n, track in enumerate(tracks, start=1):
        with reference_scope(lead.project):
            audio = track.chain.render(track.project, len(track.chain.steps) - 1, progress, cancel)
        rendered.append(audio)
        path = guard.check(target.with_name(f"{target.stem}_{n}_{_safe(track.name)}{fmt.suffix}"))
        audio_io.write_audio(path, audio, track.chain.sr, fmt, lambda f, name=track.name: progress(f"寫入 {name}", f))
        written.append(path)
        if request.dry and has_wet_effects(track.chain, track.project):
            with reference_scope(dry_lead):
                dry = track.chain.render(
                    without_wet_effects(track.chain, track.project), len(track.chain.steps) - 1, progress, cancel
                )
            path = guard.check(target.with_name(f"{target.stem}_{n}_{_safe(track.name)}{DRY_SUFFIX}{fmt.suffix}"))
            audio_io.write_audio(path, dry, track.chain.sr, fmt)
            written.append(path)
        check_cancel(cancel)
    if request.mix:
        sr = tracks[0].chain.sr
        total = max(len(a) for a in rendered)
        gains = [10.0 ** (t.gain_db / 20.0) for t in tracks]
        path = guard.check(target.with_name(f"{target.stem}{MIX_SUFFIX}{fmt.suffix}"))
        blocks = mixed_tracks_blocks(list(zip(rendered, gains)), backing, sr, total)
        audio_io.write_blocks(path, blocks, total, 2, sr, fmt, lambda f: progress("寫入混合試聽版", f))
        written.append(path)
    return written


def _safe(name: str) -> str:
    """A track name usable in a file name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "track"
