"""Keeps the audition buffers in sync with the current draft parameters.

Only the audition span (selection, or 10 s from the play head) plus padding is
recomputed when a parameter moves, so the result is audible within a second.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from voicefix.core.chain import Preview
from voicefix.core.regions import Region
from voicefix.playback.mixer import AuditionMixer, AuditionMode
from voicefix.playback.player import PlaybackError, Player
from voicefix.ui.transport import TransportBar
from voicefix.ui.waveform import WaveformView
from voicefix.ui.workers import TaskRunner

log = logging.getLogger(__name__)

PREVIEW_SECONDS = 10.0
MAX_SELECTION_PREVIEW_SECONDS = 30.0
DEBOUNCE_MS = 200
PLAYHEAD_MS = 30

PreviewFn = Callable[[Region, threading.Event], Preview]


class AuditionController(QObject):
    error = Signal(str)
    busy = Signal(bool)

    def __init__(self, sr: int, waveform: WaveformView, transport: TransportBar, runner: TaskRunner) -> None:
        super().__init__()
        self.mixer = AuditionMixer(sr)
        self.player = Player(self.mixer)
        self.waveform = waveform
        self.transport = transport
        self.runner = runner
        self.total = 0
        self.cursor = 0  # absolute play head when nothing is loaded around it
        self._span: Optional[Region] = None
        self._preview_fn: Optional[PreviewFn] = None
        self._was_playing = False
        self._debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS, timeout=self._run_preview)
        self._ticker = QTimer(self, interval=PLAYHEAD_MS, timeout=self._tick)
        self._ticker.start()

        transport.play_toggled.connect(self.toggle_play)
        transport.mode_changed.connect(self.set_mode)
        transport.loop_changed.connect(lambda on: setattr(self.mixer, "loop", on))
        transport.gain_match_changed.connect(lambda on: setattr(self.mixer, "gain_match_enabled", on))
        transport.backing_changed.connect(lambda on: setattr(self.mixer, "backing_enabled", on))
        transport.spectrogram_changed.connect(self._on_spectrogram)
        waveform.seek_requested.connect(self.seek)

    # ----- source -----------------------------------------------------------
    def reset(self, total: int, preview_fn: Optional[PreviewFn]) -> None:
        self.stop()
        self.total = total
        self.cursor = min(self.cursor, max(total - 1, 0))
        self._span = None
        self._preview_fn = preview_fn
        self.waveform.set_after(None, 0)
        self.waveform.set_preview_span(None)
        self.transport.set_gain_db(None)
        if preview_fn is not None:
            self.request()

    def set_preview_fn(self, preview_fn: PreviewFn) -> None:
        """New draft parameters: recompute the current span with them."""
        self._preview_fn = preview_fn
        self.request()

    def request(self) -> None:
        if self._preview_fn is not None:
            self._debounce.start()

    def _target_span(self) -> Region:
        selections = self.waveform.selections()
        if selections:
            first = selections[0]
            return Region(first.start, min(first.end, first.start + int(MAX_SELECTION_PREVIEW_SECONDS * self.mixer.sr)))
        if self._span is not None and self._span.start <= self.cursor < self._span.end:
            return self._span
        return Region(self.cursor, min(self.total, self.cursor + int(PREVIEW_SECONDS * self.mixer.sr)))

    def _run_preview(self) -> None:
        if self._preview_fn is None or self.total == 0:
            return
        span, fn = self._target_span(), self._preview_fn
        self.busy.emit(True)
        self.runner.start(
            lambda progress, cancel: fn(span, cancel),
            on_done=self._on_preview,
            on_failed=self._on_failed,
            on_cancelled=lambda: self.busy.emit(False),
            slot="preview",
        )

    def _on_preview(self, preview: Preview) -> None:
        self.busy.emit(False)
        self._span = preview.span
        self.mixer.set_buffers(preview.before, preview.after, preview.span.start)
        self.waveform.set_after(preview.after, preview.span.start)
        self.waveform.set_preview_span(preview.span)
        self.transport.set_gain_db(self.mixer.match_gain_db)
        self._update_spectrogram()

    def _on_failed(self, message: str) -> None:
        self.busy.emit(False)
        self.error.emit(f"試聽計算失敗：{message}")

    # ----- transport --------------------------------------------------------
    def toggle_play(self) -> None:
        if self.player.is_playing:
            self.stop()
            return
        if self.mixer.length == 0:
            return
        try:
            self.player.play()
        except PlaybackError as exc:
            self.error.emit(str(exc))
        self.transport.set_playing(self.player.is_playing)

    def stop(self) -> None:
        self.player.stop()
        self.transport.set_playing(False)

    def seek(self, sample: int) -> None:
        self.cursor = max(0, min(sample, max(self.total - 1, 0)))
        self.waveform.set_playhead(self.cursor)
        local = self.cursor - self.mixer.offset
        if self._span is not None and 0 <= local < self.mixer.length and not self.waveform.selections():
            self.mixer.seek(local)
        else:
            self.request()

    def set_mode(self, mode: AuditionMode) -> None:
        self.mixer.set_mode(mode)
        self.transport.set_mode(mode)
        self._update_spectrogram()

    def toggle_before_after(self) -> None:
        self.set_mode(AuditionMode.BEFORE if self.mixer.mode != AuditionMode.BEFORE else AuditionMode.AFTER)

    def toggle_removed(self) -> None:
        self.set_mode(AuditionMode.REMOVED if self.mixer.mode != AuditionMode.REMOVED else AuditionMode.AFTER)

    def _tick(self) -> None:
        playing = self.player.is_playing
        if playing:
            self.cursor = self.mixer.offset + self.mixer.position
            self.waveform.set_playhead(self.cursor)
        if playing != self._was_playing:  # e.g. reached the end without looping
            self.transport.set_playing(playing)
            self._was_playing = playing

    # ----- spectrogram ------------------------------------------------------
    def _on_spectrogram(self, visible: bool) -> None:
        self.waveform.set_spectrogram_visible(visible)
        self._update_spectrogram()

    def _update_spectrogram(self) -> None:
        if self.mixer.length == 0:
            self.waveform.set_spectrogram_source(None, 0)
            return
        self.waveform.set_spectrogram_source(self.mixer.signal(self.mixer.mode), self.mixer.offset)
