"""An open project: snapshot, chain, cache location and autosave."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from voicefix.core.cache import AudioCache, load_accompaniment, load_source
from voicefix.core.chain import StepChain
from voicefix.core.project import Flow, Project, load_project, save_project
from voicefix.core.regions import Region
from voicefix.steps.registry import build_steps, first_affected

log = logging.getLogger(__name__)


def default_cache_root() -> Path:
    from PySide6.QtCore import QStandardPaths

    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / "voicefix"


class Session:
    def __init__(self, project: Project, project_path: Path, cache_root: Path,
                 progress: Optional[Callable[[float], None]] = None) -> None:
        self.project_path = project_path
        self.steps = build_steps(project.flow)
        project = project.migrated([s.id for s in self.steps])
        digest = hashlib.sha1(str(project_path.resolve()).encode("utf-8")).hexdigest()[:16]
        self.cache = AudioCache(cache_root / digest)
        source, sr, key = load_source(project.source_path, self.cache, progress)
        self.chain = StepChain(self.steps, self.cache, source, sr, key)
        self.project = project
        self.backing: Optional[np.ndarray] = None
        self.backing_key: Optional[str] = None
        self.backing_error: Optional[str] = None
        if project.accompaniment_path:
            self._open_backing(project.accompaniment_path)

    @staticmethod
    def create(flow: Flow, source: Path, project_path: Path, cache_root: Path,
               progress: Optional[Callable[[float], None]] = None) -> "Session":
        steps = build_steps(flow)
        project = Project.new(flow, str(source.resolve()), [s.id for s in steps])
        session = Session(project, project_path, cache_root, progress)
        session.save()
        return session

    @staticmethod
    def open(project_path: Path, cache_root: Path, progress: Optional[Callable[[float], None]] = None) -> "Session":
        project = load_project(project_path)
        if not Path(project.source_path).exists():
            raise FileNotFoundError(f"找不到原始錄音：{project.source_path}")
        return Session(project, project_path, cache_root, progress)

    @property
    def sr(self) -> int:
        return self.chain.sr

    def update(self, project: Project) -> None:
        """Replace the snapshot and autosave immediately."""
        self.project = project
        self.save()

    def save(self) -> None:
        save_project(self.project, self.project_path)
        keep = self.chain.all_keys(self.project) | ({self.backing_key} if self.backing_key else set())
        self.cache.prune(keep)

    def go_to(self, index: int) -> None:
        self.update(replace(self.project, current_index=index))

    def set_shared(self, key: str, value: object) -> None:
        self.update(self.project.with_shared(first_affected(self.steps, key), **{key: value}))

    def _open_backing(self, path: str) -> None:
        """A broken or missing accompaniment never stops the project from opening."""
        try:
            self.backing, self.backing_key = self.prepare_accompaniment(Path(path))
            self.backing_error = None
        except Exception as exc:  # any decode failure: keep working without it
            self.backing, self.backing_key = None, None
            self.backing_error = f"無法讀取伴奏：{exc}" if Path(path).exists() else f"找不到伴奏檔：{path}"
            log.warning(self.backing_error)

    def prepare_accompaniment(self, path: Path) -> tuple[np.ndarray, str]:
        """Decode and resample into the cache (safe to run on a worker thread)."""
        return load_accompaniment(path, self.cache, self.sr, len(self.chain.source))

    def set_accompaniment(self, path: Optional[Path], prepared: Optional[tuple[np.ndarray, str]] = None) -> None:
        """Persist the choice only once the file has been read successfully. The
        accompaniment is only for listening, so no processed step becomes stale."""
        if path is not None:
            self.backing, self.backing_key = prepared or self.prepare_accompaniment(path)
        else:
            self.backing, self.backing_key = None, None
        self.backing_error = None
        self.update(self.project.with_shared(None, accompaniment_path=str(path.resolve()) if path else None))

    def to_source(self, index: int, region: Region) -> Region:
        """A region drawn on step ``index``'s input timeline, in source samples."""
        return self.chain.edit_map_before(self.project, index).region_to_source(region)
