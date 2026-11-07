"""B6 / C6 per-note targets: pick a note on the pitch plot and set which note it
should be. Needed when a note is more than half a semitone off, because the
automatic guess (the nearest in-key note) then picks the neighbour."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal

from voicefix.core.regions import Region
from voicefix.dsp.key import note_name
from voicefix.steps.base import MAX_NOTE_TARGET_CENTS, NOTE_MATCH_SECONDS
from voicefix.ui.panels import PitchPanel
from voicefix.ui.waveform import WaveformView

SEEK_BEFORE_SECONDS = 0.3


class NoteTargetEditor(QObject):
    changed = Signal()  # the targets changed: re-analyse, re-detect, re-preview
    seek = Signal(int)

    def __init__(self, panel: PitchPanel, waveform: WaveformView, sr: int,
                 to_source: Callable[[Region], Region]) -> None:
        super().__init__()
        self.panel, self.waveform, self.sr, self.to_source = panel, waveform, sr, to_source
        self._targets: dict[int, int] = {}  # source note start -> target cents
        self._notes: list[dict] = []  # every note with its target, from the analysis
        self._selected: Optional[Region] = None
        panel.shift_target.connect(self.shift)
        panel.reset_target.connect(self.reset)
        waveform.pitch_clicked.connect(self.select_at)

    def load(self, targets: dict[int, int]) -> None:
        """A new step was entered."""
        self._targets, self._notes, self._selected = dict(targets), [], None
        self.panel.show_note(None)

    def targets(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._targets.items()))

    def set_notes(self, notes: list[dict]) -> None:
        """A fresh analysis: drop targets whose note no longer exists (an earlier
        step moved or removed it) and a selection that vanished with its note."""
        self._notes = notes
        starts = [self.to_source(n["region"]).start for n in notes]
        tolerance = int(NOTE_MATCH_SECONDS * self.sr)
        self._targets = {p: c for p, c in self._targets.items() if any(abs(p - s) <= tolerance for s in starts)}
        if self._selected is not None and self._current() is None:
            self._selected = None
        self._show()

    def select_at(self, sample: int, seek: bool = True) -> None:
        if not self._notes:
            return
        inside = [n for n in self._notes if n["region"].start <= sample < n["region"].end]
        note = inside[0] if inside else min(self._notes, key=lambda n: abs(n["region"].start - sample))
        if not inside and abs(note["region"].start - sample) > self.sr // 4:
            self._selected = None  # clicked between notes
            self._show()
            return
        self._selected = note["region"]
        self._show()
        if seek:
            self.seek.emit(max(0, note["region"].start - int(SEEK_BEFORE_SECONDS * self.sr)))

    def shift(self, semitones: int) -> None:
        note = self._current()
        if note is None:
            return
        source = self.to_source(note["region"]).start
        # start from the stored target, not the last analysis: quick presses must add up
        current = self._stored(source)
        base = current if current is not None else int(round(note["target"] / 100.0)) * 100
        target = base + 100 * semitones
        if abs(target - note["cents"]) > MAX_NOTE_TARGET_CENTS:
            return
        remaining = self._without(source)
        if target == int(round(note["auto_target"] / 100.0)) * 100:
            self._targets = remaining  # back to what the program would pick anyway
        else:
            self._targets = {**remaining, source: target}
        self._show_pending(note, target)
        self.changed.emit()

    def reset(self) -> None:
        if self._selected is not None:
            self._targets = self._without(self.to_source(self._selected).start)
            self.changed.emit()

    def _current(self) -> Optional[dict]:
        if self._selected is None:
            return None
        return next((n for n in self._notes if n["region"].start == self._selected.start), None)

    def _nearest(self, source: int) -> Optional[int]:
        tolerance = int(NOTE_MATCH_SECONDS * self.sr)
        nearest = min(self._targets, key=lambda p: abs(p - source), default=None)
        return nearest if nearest is not None and abs(nearest - source) <= tolerance else None

    def _stored(self, source: int) -> Optional[int]:
        key = self._nearest(source)
        return self._targets[key] if key is not None else None

    def _without(self, source: int) -> dict[int, int]:
        """The targets minus the one belonging to this note (only the nearest)."""
        key = self._nearest(source)
        return {p: c for p, c in self._targets.items() if p != key}

    def _show_pending(self, note: dict, target: int) -> None:
        """Update the buttons right away; the full text follows the re-analysis."""
        sung = note["cents"]
        self.panel.show_note(
            f"目標改成 {note_name(target)}，重新計算中…",
            can_up=target + 100 - sung <= MAX_NOTE_TARGET_CENTS,
            can_down=sung - (target - 100) <= MAX_NOTE_TARGET_CENTS,
            manual=True,
        )

    def _show(self) -> None:
        note = self._current()
        self.waveform.set_pitch_notes(self._notes, note["region"] if note else None)
        if note is None:
            self.panel.show_note(None)
            return
        sung, target = note["cents"], note["target"]
        how = "你指定的" if note["manual"] else "程式認定的"
        detune = sung - round(sung / 100.0) * 100
        self.panel.show_note(
            f"唱的是 {note_name(sung)}（{detune:+.0f}），{how}目標是 {note_name(target)}，偏 {sung - target:+.0f} 音分。",
            can_up=target + 100 - sung <= MAX_NOTE_TARGET_CENTS,
            can_down=sung - (target - 100) <= MAX_NOTE_TARGET_CENTS,
            manual=note["manual"],
        )
