"""Render the whole chain and write the result(s) to disk."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional

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

    def __init__(self, project: Project) -> None:
        inputs = [project.source_path, project.accompaniment_path]
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
