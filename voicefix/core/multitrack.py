"""Flow C project: one lead and up to four harmony / double tracks.

Every track keeps a complete per-track ``Project`` so the single-track engine
(chain, cache, steps) runs unchanged. Sharing is a layer on top:

* step status and the current step are always the same on every track;
* a step's parameters (strength, option, scope, advanced, settings) are shared
  by every track that is linked for that step; a track can be unlinked for one
  step to get its own values;
* selections, ticked events, protected regions and the noise sample are always
  per track (they are positions in that track's own audio).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from voicefix.core.project import FORMAT_VERSION, Flow, Project

MAX_TRACKS = 5
LEAD = 0
DEFAULT_HARMONY_GAIN_DB = -3.0


@dataclass(frozen=True)
class Track:
    name: str
    project: Project


@dataclass(frozen=True)
class MultiProject:
    tracks: tuple[Track, ...]
    unlinked: frozenset[tuple[int, str]] = frozenset()  # (track, step id) with its own values
    accompaniment_path: Optional[str] = None
    harmony_gain_db: float = DEFAULT_HARMONY_GAIN_DB  # level of the other tracks in the mix preview
    current_track: int = LEAD

    def __post_init__(self) -> None:
        if not 1 <= len(self.tracks) <= MAX_TRACKS:
            raise ValueError(f"流程 C 需要 1～{MAX_TRACKS} 軌（目前 {len(self.tracks)} 軌）")

    @staticmethod
    def new(names: list[str], sources: list[str], step_ids: list[str]) -> "MultiProject":
        tracks = tuple(
            Track(name, Project.new(Flow.MULTITRACK, source, step_ids)) for name, source in zip(names, sources)
        )
        return MultiProject(tracks)

    @property
    def step_ids(self) -> tuple[str, ...]:
        return self.tracks[LEAD].project.step_ids

    def project(self, track: int) -> Project:
        return self.tracks[track].project

    def is_linked(self, track: int, step_id: str) -> bool:
        return (track, step_id) not in self.unlinked

    def with_linked(self, track: int, step_id: str, linked: bool) -> "MultiProject":
        unlinked = set(self.unlinked)
        (unlinked.discard if linked else unlinked.add)((track, step_id))
        updated = replace(self, unlinked=frozenset(unlinked))
        if linked:  # re-joining: take the values the other linked tracks use
            donor = next((t for t in range(len(self.tracks)) if t != track and updated.is_linked(t, step_id)), None)
            if donor is not None:
                updated = updated.propagate(donor, updated.project(donor))
        return updated

    def propagate(self, source: int, project: Project) -> "MultiProject":
        """Store ``project`` as track ``source`` and share what is shared."""
        tracks = []
        for t, track in enumerate(self.tracks):
            if t == source:
                tracks.append(replace(track, project=project))
                continue
            steps = dict(track.project.steps)
            for sid in project.step_ids:
                theirs, ours = steps[sid], project.steps[sid]
                shared = {"status": ours.status}
                if self.is_linked(source, sid) and self.is_linked(t, sid):
                    shared.update(
                        strength=ours.strength, option=ours.option, scope=ours.scope,
                        advanced=ours.advanced, settings=ours.settings,
                    )
                steps[sid] = replace(theirs, **shared)
            tracks.append(replace(track, project=replace(track.project, steps=steps, current_index=project.current_index)))
        return replace(self, tracks=tuple(tracks))

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "flow": Flow.MULTITRACK.value,
            "accompaniment_path": self.accompaniment_path,
            "harmony_gain_db": self.harmony_gain_db,
            "current_track": self.current_track,
            "unlinked": sorted([t, sid] for t, sid in self.unlinked),
            "tracks": [{"name": t.name, "project": t.project.to_json()} for t in self.tracks],
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "MultiProject":
        if data.get("format_version") != FORMAT_VERSION or data.get("flow") != Flow.MULTITRACK.value:
            raise ValueError("這不是流程 C 的多軌專案檔")
        return MultiProject(
            tracks=tuple(Track(t["name"], Project.from_json(t["project"])) for t in data["tracks"]),
            unlinked=frozenset((int(t), str(sid)) for t, sid in data.get("unlinked", [])),
            accompaniment_path=data.get("accompaniment_path"),
            harmony_gain_db=float(data.get("harmony_gain_db", DEFAULT_HARMONY_GAIN_DB)),
            current_track=int(data.get("current_track", LEAD)),
        )


def is_multitrack_file(path: str | Path) -> bool:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("flow") == Flow.MULTITRACK.value
    except (OSError, ValueError):
        return False


def save_multiproject(multi: MultiProject, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(multi.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_multiproject(path: str | Path) -> MultiProject:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法開啟專案檔：{exc}") from exc
    return MultiProject.from_json(data)

