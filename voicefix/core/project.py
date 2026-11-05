"""Project model: immutable snapshots, JSON persistence, step status rules.

A project records the source path and every step's parameters, never audio.
All regions are stored in source-timeline samples (see ``editmap``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from voicefix.core.regions import Region, ScopeMode

FORMAT_VERSION = 1
PROJECT_SUFFIX = ".voicefix"


class Flow(str, Enum):
    VOICE_WORK = "A"
    SINGING = "B"


FLOW_LABELS = {Flow.VOICE_WORK: "流程 A：音聲作品", Flow.SINGING: "流程 B：歌聲"}


class StepStatus(str, Enum):
    NOT_STARTED = "not_started"
    APPLIED = "applied"
    SKIPPED = "skipped"
    STALE = "stale"

    @property
    def is_done(self) -> bool:
        return self in (StepStatus.APPLIED, StepStatus.SKIPPED, StepStatus.STALE)

    @property
    def is_active(self) -> bool:
        """Whether the step's processing contributes to the output."""
        return self in (StepStatus.APPLIED, StepStatus.STALE)


STATUS_LABELS = {
    StepStatus.NOT_STARTED: "未開始",
    StepStatus.APPLIED: "已套用",
    StepStatus.SKIPPED: "已略過",
    StepStatus.STALE: "需重算",
}


@dataclass(frozen=True)
class StepConfig:
    strength: Optional[int] = None
    option: Optional[str] = None
    scope: ScopeMode = ScopeMode.WHOLE
    selections: tuple[Region, ...] = ()
    disabled_events: tuple[int, ...] = ()  # source-timeline start samples
    advanced: Mapping[str, float] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)  # step-specific, JSON-safe
    status: StepStatus = StepStatus.NOT_STARTED

    def fingerprint(self) -> dict[str, Any]:
        """Everything that influences processing (status excluded)."""
        data = self.to_json()
        data.pop("status")
        return data

    def to_json(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "option": self.option,
            "scope": self.scope.value,
            "selections": [r.to_json() for r in self.selections],
            "disabled_events": list(self.disabled_events),
            "advanced": dict(sorted(self.advanced.items())),
            "settings": _sorted_json(self.settings),
            "status": self.status.value,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "StepConfig":
        return StepConfig(
            strength=data.get("strength"),
            option=data.get("option"),
            scope=ScopeMode(data.get("scope", ScopeMode.WHOLE.value)),
            selections=tuple(Region.from_json(r) for r in data.get("selections", [])),
            disabled_events=tuple(int(x) for x in data.get("disabled_events", [])),
            advanced={k: float(v) for k, v in data.get("advanced", {}).items()},
            settings=dict(data.get("settings", {})),
            status=StepStatus(data.get("status", StepStatus.NOT_STARTED.value)),
        )


def _sorted_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


@dataclass(frozen=True)
class Project:
    flow: Flow
    source_path: str
    step_ids: tuple[str, ...]
    steps: Mapping[str, StepConfig]
    protected: tuple[Region, ...] = ()
    noise_sample: Optional[Region] = None
    binaural: bool = False
    accompaniment_path: Optional[str] = None
    current_index: int = 0

    @staticmethod
    def new(flow: Flow, source_path: str, step_ids: Sequence[str]) -> "Project":
        return Project(
            flow=flow,
            source_path=str(source_path),
            step_ids=tuple(step_ids),
            steps={sid: StepConfig() for sid in step_ids},
        )

    def config(self, index: int) -> StepConfig:
        return self.steps[self.step_ids[index]]

    def status(self, index: int) -> StepStatus:
        return self.config(index).status

    def is_unlocked(self, index: int) -> bool:
        return all(self.status(i).is_done for i in range(index))

    def with_config(self, index: int, config: StepConfig) -> "Project":
        steps = dict(self.steps)
        steps[self.step_ids[index]] = config
        return replace(self, steps=steps)

    def _commit(self, index: int, config: StepConfig) -> "Project":
        """Store a committed config and mark later applied steps as stale."""
        steps = dict(self.steps)
        steps[self.step_ids[index]] = config
        for sid in self.step_ids[index + 1 :]:
            if steps[sid].status == StepStatus.APPLIED:
                steps[sid] = replace(steps[sid], status=StepStatus.STALE)
        return replace(self, steps=steps)

    def apply_step(self, index: int, config: StepConfig) -> "Project":
        return self._commit(index, replace(config, status=StepStatus.APPLIED))

    def skip_step(self, index: int) -> "Project":
        return self._commit(index, replace(self.config(index), status=StepStatus.SKIPPED))

    def mark_fresh(self, index: int) -> "Project":
        """A stale step was recomputed with unchanged parameters."""
        if self.status(index) != StepStatus.STALE:
            return self
        return self.with_config(index, replace(self.config(index), status=StepStatus.APPLIED))

    def with_shared(self, first_affected: Optional[int], **changes: Any) -> "Project":
        """Change a project-wide setting (protected regions, noise sample, binaural).

        Applied steps from ``first_affected`` onward become stale; ``None`` means
        no step's output depends on the change.
        """
        updated = replace(self, **changes)
        if first_affected is None:
            return updated
        steps = dict(updated.steps)
        for sid in self.step_ids[first_affected:]:
            if steps[sid].status == StepStatus.APPLIED:
                steps[sid] = replace(steps[sid], status=StepStatus.STALE)
        return replace(updated, steps=steps)

    def shared_fingerprint(self, keys: Sequence[str] = ("protected", "noise_sample", "binaural")) -> dict[str, Any]:
        full = {
            "protected": [r.to_json() for r in self.protected],
            "noise_sample": self.noise_sample.to_json() if self.noise_sample else None,
            "binaural": self.binaural,
        }
        return {k: full[k] for k in keys}

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "flow": self.flow.value,
            "source_path": self.source_path,
            "accompaniment_path": self.accompaniment_path,
            "current_index": self.current_index,
            **self.shared_fingerprint(("protected", "noise_sample", "binaural")),
            "steps": [{"id": sid, **self.steps[sid].to_json()} for sid in self.step_ids],
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "Project":
        version = data.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"不支援的專案檔版本：{version}")
        step_entries = data["steps"]
        noise = data.get("noise_sample")
        return Project(
            flow=Flow(data["flow"]),
            source_path=data["source_path"],
            step_ids=tuple(entry["id"] for entry in step_entries),
            steps={entry["id"]: StepConfig.from_json(entry) for entry in step_entries},
            protected=tuple(Region.from_json(r) for r in data.get("protected", [])),
            noise_sample=Region.from_json(noise) if noise else None,
            binaural=bool(data.get("binaural", False)),
            accompaniment_path=data.get("accompaniment_path"),
            current_index=int(data.get("current_index", 0)),
        )


def save_project(project: Project, path: str | Path) -> None:
    """Atomic write: a crash mid-save never leaves a truncated project file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(project.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_project(path: str | Path) -> Project:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法開啟專案檔：{exc}") from exc
    return Project.from_json(data)
