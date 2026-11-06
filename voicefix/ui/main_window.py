"""One screen per step: header, waveform, controls, audition bar.

Every step runs the same loop: analyse -> suggest -> adjust -> listen -> apply.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from voicefix.core.exporter import ExportRequest, has_wet_effects
from voicefix.core.project import Flow, StepConfig, StepStatus
from voicefix.core.regions import Region, merge_regions
from voicefix.steps.base import Analysis, Event, Step
from voicefix.ui.audition import AuditionController
from voicefix.ui.controls import EventList, StepControls
from voicefix.ui.panels import EditPanel, ExportPanel, ImportPanel, PitchPanel
from voicefix.ui.session import Session
from voicefix.ui.start_dialog import AUDIO_FILTER
from voicefix.ui.step_list import StepList
from voicefix.ui.trackbar import TrackBar
from voicefix.ui.transport import TransportBar
from voicefix.ui.waveform import WaveformView
from voicefix.ui.workers import TaskRunner

log = logging.getLogger(__name__)

BADGE_RUN = "background:#27ae60;color:white;padding:2px 8px;border-radius:8px;"
BADGE_SKIP = "background:#7f8c8d;color:white;padding:2px 8px;border-radius:8px;"


class MainWindow(QMainWindow):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.runner = TaskRunner(self)
        self.index = -1
        self.events: list[Event] = []
        self.step_settings: dict[str, Any] = {}
        self._long_task = None
        self._user_disabled: tuple[int, ...] = ()
        self._input_ready = False
        self._last_analysis: Optional[Analysis] = None
        self.setWindowTitle(f"VoiceFix — {Path(session.project.source_path).name}")
        self.resize(1440, 860)
        self._build_ui()
        self.audition = AuditionController(session.sr, self.waveform, self.transport, self.runner)
        self._others = None  # flow C: the other tracks' input to this step, summed
        self.audition.error.connect(self.show_error)
        self.audition.busy.connect(lambda busy: self.statusBar().showMessage("試聽計算中…" if busy else "", 0))
        self._connect()
        self.enter_step(self._initial_index())

    # ================================================================ layout
    def _build_ui(self) -> None:
        self.step_list = StepList()
        self.title = QLabel(styleSheet="font-size: 20px; font-weight: bold;")
        self.badge = QLabel()
        self.description = QLabel(wordWrap=True, styleSheet="font-size: 14px;")
        self.term = QLabel(styleSheet="color: gray; font-size: 11px;")
        self.summary = QLabel(wordWrap=True, styleSheet="color: #555;")
        title_row = QHBoxLayout()
        title_row.addWidget(self.title)
        title_row.addWidget(self.badge)
        title_row.addStretch(1)
        title_row.addWidget(self.term)
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.addLayout(title_row)
        self.track_bar = TrackBar()
        self.track_bar.setVisible(self.session.is_multitrack)
        header_layout.addWidget(self.track_bar)
        header_layout.addWidget(self.description)
        header_layout.addWidget(self.summary)

        self.protect_button = QPushButton("把選取標成保護區段")
        self.protect_button.setToolTip("刻意的表演音（舔耳、耳語…）。A4～A8 不會處理保護區段。")
        self.unprotect_button = QPushButton("移除保護區段")
        self.clear_selection_button = QPushButton("清除選取")
        self.zoom_button = QPushButton("放大到選取")
        tools = QHBoxLayout()
        for button in (self.protect_button, self.unprotect_button, self.clear_selection_button, self.zoom_button):
            tools.addWidget(button)
        tools.addStretch(1)
        tools.addWidget(QLabel("拖曳＝選取　Ctrl＋拖曳＝多選　滾輪＝縮放　右鍵拖曳＝平移", styleSheet="color: gray;"))
        self.waveform = WaveformView()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(header)
        center_layout.addLayout(tools)
        center_layout.addWidget(self.waveform, 1)

        self.controls = StepControls()
        self.event_list = EventList()
        self.import_panel = ImportPanel()
        self.edit_panel = EditPanel()
        self.export_panel = ExportPanel()
        self.pitch_panel = PitchPanel()
        right = QWidget()
        right_layout = QVBoxLayout(right)
        panels = (self.import_panel, self.edit_panel, self.pitch_panel, self.controls, self.event_list, self.export_panel)
        for widget in panels:
            right_layout.addWidget(widget)
        right_layout.addStretch(1)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(right)
        self._edit_widgets = (
            self.protect_button, self.unprotect_button, self.clear_selection_button, self.zoom_button,
            self.waveform, right,
        )
        scroll.setMinimumWidth(300)

        splitter = QSplitter()
        splitter.addWidget(self.step_list)
        splitter.addWidget(center)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 860, 340])

        self.transport = TransportBar()
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self.transport)
        self.setCentralWidget(root)

        self.progress = QProgressBar(maximum=1000, maximumWidth=260, visible=False)
        self.cancel_button = QPushButton("取消", visible=False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)

    def _connect(self) -> None:
        self.step_list.step_chosen.connect(self._on_step_chosen)
        self.controls.changed.connect(self._on_controls_changed)
        self.controls.reset_requested.connect(self._reset_to_suggestion)
        self.event_list.toggled.connect(self._on_events_toggled)
        self.event_list.activated.connect(self._jump_to_event)
        self.import_panel.binaural_changed.connect(lambda on: self._set_shared("binaural", on))
        self.import_panel.use_selection_as_noise.connect(self._use_selection_as_noise)
        self.import_panel.show_noise.connect(self._show_noise)
        self.track_bar.track_chosen.connect(self._choose_track)
        self.track_bar.own_settings_changed.connect(self._set_own_settings)
        self.transport.backing_changed.connect(lambda _: self._refresh_backing())
        self.transport.others_changed.connect(lambda _: self._refresh_backing())
        self.export_panel.harmony_gain.valueChanged.connect(self._set_harmony_gain)
        self.import_panel.choose_accompaniment.connect(self._choose_accompaniment)
        self.import_panel.clear_accompaniment.connect(lambda: self._set_accompaniment(None))
        self.import_panel.key_changed.connect(self._set_key)
        self.pitch_panel.key_changed.connect(self._set_pitch_key)
        self.edit_panel.edit_requested.connect(self._add_edit)
        self.edit_panel.remove_requested.connect(self._remove_edit)
        self.export_panel.export_requested.connect(self._export)
        self.protect_button.clicked.connect(self._protect_selection)
        self.unprotect_button.clicked.connect(self._unprotect)
        self.clear_selection_button.clicked.connect(self.waveform.clear_selections)
        self.zoom_button.clicked.connect(self._zoom_to_selection)
        self.waveform.selections_changed.connect(self._replan)
        self.transport.previous_requested.connect(lambda: self.enter_step(self.index - 1))
        self.transport.skip_requested.connect(self._skip)
        self.transport.apply_requested.connect(self._apply)
        self.cancel_button.clicked.connect(self._cancel_long_task)
        for key, slot in (
            ("Space", self.audition.toggle_play),
            ("B", self.audition.toggle_before_after),
            ("R", self.audition.toggle_removed),
        ):
            QShortcut(QKeySequence(key), self, activated=slot)

    # ============================================================ properties
    @property
    def project(self):
        return self.session.project

    @property
    def chain(self):
        return self.session.chain

    @property
    def step(self) -> Step:
        return self.session.steps[self.index]

    def _initial_index(self) -> int:
        wanted = self.project.current_index
        if 0 <= wanted < len(self.session.steps) and self.project.is_unlocked(wanted):
            return wanted
        return next((i for i in range(len(self.session.steps)) if not self.project.status(i).is_done), 0)

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, "VoiceFix", message)

    def _refresh_step_list(self) -> None:
        self.step_list.refresh(self.session.steps, self.project, self.index)

    # ========================================================== step entry
    def _on_step_chosen(self, index: int) -> None:
        if index != self.index and self.project.is_unlocked(index) and self._long_task is None:
            self.enter_step(index)
        else:
            self._refresh_step_list()

    def enter_step(self, index: int) -> None:
        if not 0 <= index < len(self.session.steps):
            return
        self.runner.cancel_all()
        self.index = index
        self.session.go_to(index)
        step = self.step
        config = self.project.config(index)
        self.step_settings = dict(config.settings)
        if step.panel == "pitch" and "key" not in self.step_settings:
            # start from the key confirmed in B0 (or auto-detect)
            self.step_settings["key"] = self.project.config(0).settings.get("key", "auto")
        self.events = []
        self._user_disabled = tuple(config.disabled_events)  # source positions the user unticked
        self._last_analysis = None
        self._input_ready = False
        self._set_editing(False)  # the waveform still shows the previous step's timeline

        self.title.setText(f"{step.code}　{step.title}")
        self.description.setText(step.description)
        self.term.setText(step.term)
        self.badge.setText("")
        self.badge.setStyleSheet("")
        self.summary.setText("分析中…")
        self.controls.load(step, config)
        self.controls.setVisible(step.has_strength or bool(step.options) or len(step.scopes) > 1)
        self.event_list.setVisible(step.event_based)
        singing = self.project.flow in (Flow.SINGING, Flow.MULTITRACK)
        multitrack = self.session.is_multitrack
        self.import_panel.setVisible(step.panel == "import")
        self.import_panel.configure(singing, self.session.track_names)
        self.edit_panel.setVisible(step.changes_length)
        self.pitch_panel.setVisible(step.panel == "pitch")
        self.waveform.set_pitch(None, 1)
        self.export_panel.setVisible(step.panel == "export")
        gain = self.session.multi.harmony_gain_db if multitrack else 0.0
        self.export_panel.configure(
            singing, has_wet_effects(self.chain, self.project), self.session.backing is not None, multitrack, gain
        )
        self.transport.set_backing(self.session.backing is not None, step.audition_with_backing)
        self.transport.set_others(multitrack, step.audition_with_backing)
        if multitrack:
            has_settings = step.has_strength or bool(step.options) or bool(step.advanced)
            self.track_bar.refresh(
                self.session.track_names, self.session.current_track, not self.session.is_linked(step.id), has_settings
            )
        self._others = None
        self._refresh_backing()
        is_last = index == len(self.session.steps) - 1
        self.transport.set_navigation(index > 0, not is_last, None if is_last else "套用並前往下一步 ▶")
        self._refresh_step_list()
        self.audition.reset(0, None)
        self._ensure_input(self._on_input_ready)

    def _ensure_input(self, then: Callable[[], None]) -> None:
        """Render the previous step's output first if it is stale or missing."""
        upto = self.index - 1
        if self.session.is_ready(upto):
            then()
            return
        session = self.session
        self._run_long(
            "準備這一步的輸入（重算前面的步驟）",
            lambda progress, cancel: session.render_upto(upto, progress, cancel),
            lambda _: (self._mark_fresh(upto), then()),
            on_cancelled=lambda: self.summary.setText("已取消。請重新選擇步驟。"),
        )

    def _mark_fresh(self, upto: int) -> None:
        """Steps up to ``upto`` were just rendered. Any later stale step whose output
        is already cached (e.g. after re-applying a passthrough step with unchanged
        settings) is not really stale either."""
        project = self.project
        for i in range(len(self.session.steps)):
            if i <= upto or (project.status(i) == StepStatus.STALE and self.chain.is_ready(project, i)):
                project = project.mark_fresh(i)
        self.session.update(project)
        self._refresh_step_list()

    def _set_editing(self, enabled: bool) -> None:
        """Region edits are only safe once this step's input is on screen and no
        background render could overwrite the project."""
        for widget in self._edit_widgets:
            widget.setEnabled(enabled)

    def _on_input_ready(self) -> None:
        self._input_ready = True
        self._set_editing(self._long_task is None)
        audio = self.chain.render(self.project, self.index - 1)
        self.waveform.set_input(audio, self.session.sr, reset_view=True)
        ctx = self.chain.context(self.project, self.index)  # stored config, not the previous screen's draft
        self.waveform.set_selections(ctx.selections)
        self.waveform.set_protected(ctx.protected)
        self.waveform.set_marks([])
        self.waveform.set_events([], [])
        self._refresh_panels()
        self._start_analysis()
        if self.step.event_based:
            self._detect_events()
        self.audition.reset(len(audio), self._preview_fn())
        if self.session.is_multitrack:
            self._load_others()

    def _start_analysis(self) -> None:
        project, index, draft = self.project, self.index, self._draft()
        self.runner.start(
            lambda progress, cancel: self.chain.analyze(project, index, cancel, draft),
            on_done=self._on_analysis,
            on_failed=lambda msg: self.summary.setText(f"分析失敗：{msg}"),
            slot="analysis",
        )

    def _on_analysis(self, analysis: Analysis) -> None:
        self.summary.setText("\n".join(analysis.summary))
        self.badge.setText("建議執行" if analysis.recommend else "可略過")
        self.badge.setStyleSheet(BADGE_RUN if analysis.recommend else BADGE_SKIP)
        self.waveform.set_marks(analysis.marks)
        self._last_analysis = analysis
        draft = self.controls.draft  # only fill in suggestions once, never over the user's own values
        if self.project.status(self.index) == StepStatus.NOT_STARTED and draft.strength is None and draft.option is None:
            self._apply_suggestion(analysis)
        if "pitch" in analysis.data:
            track = analysis.data["pitch"]
            self.waveform.set_pitch(track.cents, track.frame)
        if self.step.panel in ("import", "pitch"):
            self._refresh_panels()  # shows the detected key
        if self.step.panel == "import" and self.project.noise_sample is None and analysis.data.get("noise_sample"):
            self._set_shared("noise_sample", analysis.data["noise_sample"])
            self._show_noise()

    def _apply_suggestion(self, analysis: Analysis) -> None:
        draft = replace(
            self.controls.draft,
            strength=analysis.suggested_strength if self.step.has_strength else None,
            option=analysis.suggested_option or self.step.default_option,
            advanced={},
        )
        self.controls.load(self.step, draft)
        self._on_controls_changed()

    def _reset_to_suggestion(self) -> None:
        if self._last_analysis is not None:
            self._apply_suggestion(self._last_analysis)

    # =============================================================== drafts
    def _draft(self) -> StepConfig:
        """Current on-screen parameters as a StepConfig (regions in source samples)."""
        base = self.controls.draft
        selections = tuple(self.session.to_source(self.index, r) for r in self.waveform.selections())
        return replace(
            base, selections=selections, disabled_events=self._user_disabled, settings=dict(self.step_settings)
        )

    def _preview_fn(self):
        project, index, draft = self.project, self.index, self._draft()
        return lambda span, cancel: self.chain.preview(project, index, draft, span, cancel)

    def _replan(self) -> None:
        """Parameters changed: refresh the preview function and recompute."""
        if self.waveform.has_input:
            self.audition.set_preview_fn(self._preview_fn())

    def _on_controls_changed(self) -> None:
        if self.step.event_based:
            self._detect_events()
        self._replan()

    # =============================================================== events
    def _detect_events(self) -> None:
        project, index, draft = self.project, self.index, self._draft()
        self.event_list.set_busy("偵測中…")
        self.runner.start(
            lambda progress, cancel: self.chain.events(project, index, draft, cancel),
            on_done=lambda events: self._on_events(events, draft),
            on_failed=lambda msg: self.event_list.set_busy(f"偵測失敗：{msg}"),
            slot="events",
        )

    def _on_events(self, events: list[Event], draft: StepConfig) -> None:
        self.events = events
        ctx = self.chain.context(self.project, self.index, draft)
        protected = ctx.protected
        enabled = [not ctx.is_disabled(e.region) and not any(e.region.overlaps(p) for p in protected) for e in events]
        self.event_list.set_events(events, enabled, self.session.sr, self.step.event_label)
        self.waveform.set_events([e.region for e in events], enabled)

    def _on_events_toggled(self) -> None:
        edit_map = self.chain.edit_map_before(self.project, self.index)
        protected = self.chain.context(self.project, self.index).protected
        # events switched off only because they sit in a protected region are not user choices
        self._user_disabled = tuple(
            edit_map.to_source(e.region.start)
            for e, on in zip(self.events, self.event_list.enabled())
            if not on and not any(e.region.overlaps(p) for p in protected)
        )
        self.waveform.set_events([e.region for e in self.events], self.event_list.enabled())
        self._replan()

    def _jump_to_event(self, row: int) -> None:
        if 0 <= row < len(self.events):
            region = self.events[row].region
            self.waveform.zoom_to(region)
            self.audition.seek(max(0, region.start - int(0.5 * self.session.sr)))

    # ======================================================= shared settings
    def _set_shared(self, key: str, value: object) -> None:
        self.session.set_shared(key, value)
        self._refresh_panels()
        self._refresh_step_list()

    def _refresh_panels(self) -> None:
        sr = self.session.sr
        if self.step.panel == "import":
            self.import_panel.set_state(self.project.binaural, self.project.noise_sample, sr)
            detected = self._last_analysis.data.get("key") if self._last_analysis else None
            self.import_panel.set_song_state(
                self.project.accompaniment_path, self.session.backing_error, self.step_settings.get("key", "auto"), detected
            )
        if self.step.changes_length:
            self.edit_panel.set_edits(self.step_settings, sr)
        if self.step.panel == "pitch":
            detected = self._last_analysis.data.get("estimated_key") if self._last_analysis else None
            self.pitch_panel.set_key(self.step_settings.get("key", "auto"), detected)

    def _use_selection_as_noise(self) -> None:
        selections = self.waveform.selections()
        if not selections:
            self.show_error("請先在波形上拖曳選取一段只有背景聲的區段。")
            return
        self._set_shared("noise_sample", self.session.to_source(self.index, selections[0]))

    def _show_noise(self) -> None:
        if self.project.noise_sample:
            local = self.chain.edit_map_before(self.project, self.index).region_to_edited(self.project.noise_sample)
            if local:
                self.waveform.set_selections([local])
                self.waveform.zoom_to(local)
                self._replan()

    def _protect_selection(self) -> None:
        selections = [self.session.to_source(self.index, r) for r in self.waveform.selections()]
        if not selections:
            self.show_error("請先在波形上拖曳選取要保護的區段。")
            return
        self._set_shared("protected", tuple(merge_regions([*self.project.protected, *selections])))
        self._after_protection_change()

    def _unprotect(self) -> None:
        selections = [self.session.to_source(self.index, r) for r in self.waveform.selections()]
        if selections:
            kept = tuple(p for p in self.project.protected if not any(p.overlaps(s) for s in selections))
        else:
            answer = QMessageBox.question(self, "VoiceFix", "沒有選取任何區段。要移除全部保護區段嗎？")
            kept = () if answer == QMessageBox.StandardButton.Yes else self.project.protected
        self._set_shared("protected", kept)
        self._after_protection_change()

    def _after_protection_change(self) -> None:
        ctx = self.chain.context(self.project, self.index)
        self.waveform.set_protected(ctx.protected)
        if self.step.event_based and self.events:
            self._on_events(self.events, self._draft())
        self._replan()

    def _zoom_to_selection(self) -> None:
        selections = self.waveform.selections()
        if selections:
            self.waveform.zoom_to(Region(selections[0].start, selections[-1].end))

    # ================================================================= A1
    def _add_edit(self, kind: str) -> None:
        selections = self.waveform.selections()
        if not selections:
            self.show_error("請先在波形上拖曳選取要處理的片段。")
            return
        existing = [Region.from_json(r) for r in self.step_settings.get(kind, [])]
        merged = merge_regions([*existing, *(self.session.to_source(self.index, r) for r in selections)])
        self.step_settings = {**self.step_settings, kind: [r.to_json() for r in merged]}
        self.waveform.clear_selections(emit=False)
        self._refresh_panels()
        self._replan()

    def _remove_edit(self, kind: str, index: int) -> None:
        items = list(self.step_settings.get(kind, []))
        if 0 <= index < len(items):
            del items[index]
            self.step_settings = {**self.step_settings, kind: items}
            self._refresh_panels()
            self._replan()

    # ========================================================= apply / skip
    def _apply(self) -> None:
        candidate = self.project.apply_step(self.index, self._draft())
        index = self.index
        if self.step.passthrough:
            self._finish_apply(candidate, index)
            return
        self.audition.stop()
        session = self.session
        self._run_long(
            f"套用 {self.step.code} {self.step.title}" + ("（所有軌）" if session.is_multitrack else ""),
            lambda progress, cancel: session.render_step(candidate, index, progress, cancel),
            lambda _: self._finish_apply(candidate, index),
        )

    def _finish_apply(self, project, index: int) -> None:
        self.session.update(project)
        self._mark_fresh(index)
        self.enter_step(min(index + 1, len(self.session.steps) - 1))

    def _skip(self) -> None:
        self.session.update(self.project.skip_step(self.index))
        self.enter_step(self.index + 1)

    # =============================================================== export
    def _export(self, format_key: str, split: bool, dry: bool = False, mix: bool = False) -> None:
        source = Path(self.project.source_path)
        suggested = str(source.with_name(f"{source.stem}_clean"))
        path, _ = QFileDialog.getSaveFileName(self, "匯出", suggested)
        if not path:
            return
        # export options are deliberately not step settings: changing them must not
        # invalidate B12's rendered output
        project = self.project.apply_step(self.index, self._draft())  # B12 also commits its loudness preset
        self.session.update(project)
        self._refresh_step_list()
        request = ExportRequest(Path(path), format_key, split, dry, mix)
        session = self.session
        self.audition.stop()
        self._run_long(
            "匯出",
            lambda progress, cancel: session.export(project, request, progress, cancel),
            self._on_exported,
        )

    # ======================================================== accompaniment
    def _choose_accompaniment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇伴奏", str(Path(self.project.source_path).parent), AUDIO_FILTER)
        if path:
            self._set_accompaniment(Path(path))

    def _set_accompaniment(self, path: Optional[Path]) -> None:
        if path is None:
            self._accompaniment_ready(None, None)
            return
        session = self.session
        self.audition.stop()
        self._run_long(
            "讀取伴奏",
            lambda progress, cancel: session.prepare_accompaniment(path),
            lambda prepared: self._accompaniment_ready(path, prepared),
        )

    def _accompaniment_ready(self, path: Optional[Path], prepared) -> None:
        self.session.set_accompaniment(path, prepared)
        self.transport.set_backing(self.session.backing is not None, self.step.audition_with_backing)
        self._refresh_backing()
        self._refresh_panels()

    # ============================================================ listening
    def _refresh_backing(self) -> None:
        """What plays along with the track under test: accompaniment and/or the other tracks."""
        parts = []
        if self.transport.backing.isChecked() and self.session.backing is not None:
            parts.append(self.session.backing)
        if self.transport.others.isChecked() and self._others is not None:
            parts.append(self._others)
        combined = None
        if parts:
            combined = parts[0] if len(parts) == 1 else (np.asarray(parts[0], dtype=np.float32) + parts[1])
        self.audition.mixer.set_backing(combined)
        self.audition.mixer.backing_enabled = combined is not None

    def _load_others(self) -> None:
        session, index, track = self.session, self.index, self.session.current_track
        self.runner.start(
            lambda progress, cancel: (track, index, session.others_mix(index, exclude=track)),
            on_done=self._others_ready,
            on_failed=lambda msg: self.show_error(f"無法準備其他軌：{msg}"),
            slot="others",
        )

    def _others_ready(self, result) -> None:
        track, index, mixed = result
        if (track, index) != (self.session.current_track, self.index):
            return  # the user moved to another track or step meanwhile
        self._others = mixed
        self._refresh_backing()

    # ============================================================ tracks (C)
    def _choose_track(self, track: int) -> None:
        if self._long_task is not None or track == self.session.current_track:
            return
        self.session.current_track = track
        self.enter_step(self.index)

    def _set_own_settings(self, own: bool) -> None:
        """Unlinking keeps the current values; re-linking takes the shared ones back."""
        self.session.set_linked(self.step.id, not own)
        self.enter_step(self.index)

    def _set_harmony_gain(self, gain_db: float) -> None:
        if self.session.is_multitrack:
            self.session.set_harmony_gain(gain_db)

    def _set_pitch_key(self, key: str) -> None:
        """B6's key changes which notes count as off-key: re-detect and re-preview."""
        self.step_settings = {**self.step_settings, "key": key}
        self._start_analysis()  # the summary names the key in use
        self._detect_events()
        self._replan()

    def _set_key(self, key: str) -> None:
        """Saved immediately and without changing the step's status: the key does
        not affect any processing yet (it is for pitch correction, v1)."""
        self.step_settings = {**self.step_settings, "key": key}
        config = self.project.config(self.index)
        self.session.update(self.project.with_config(self.index, replace(config, settings=self.step_settings)))

    def _on_exported(self, paths: list[Path]) -> None:
        self._mark_fresh(len(self.session.steps) - 1)
        listing = "\n".join(str(p) for p in paths[:10]) + ("\n…" if len(paths) > 10 else "")
        self.export_panel.status.setText(f"已匯出 {len(paths)} 個檔案。")
        QMessageBox.information(self, "匯出完成", f"已匯出 {len(paths)} 個檔案：\n{listing}")

    # ========================================================= long tasks
    def _run_long(self, title: str, job, on_done, on_cancelled: Optional[Callable[[], None]] = None) -> None:
        self.transport.set_busy(True)
        self._set_editing(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.statusBar().showMessage(title)

        def finish() -> None:
            self._long_task = None
            self.transport.set_busy(False)
            self._set_editing(self._input_ready)
            self.progress.setVisible(False)
            self.cancel_button.setVisible(False)
            self.statusBar().clearMessage()

        def done(result: Any) -> None:
            finish()
            on_done(result)

        def failed(message: str) -> None:
            finish()
            self.show_error(f"{title}失敗：{message}")

        def cancelled() -> None:
            finish()
            self.statusBar().showMessage("已取消", 3000)
            if on_cancelled:
                on_cancelled()

        def progress(message: str, fraction: float) -> None:
            self.statusBar().showMessage(message)
            self.progress.setValue(int(fraction * 1000))

        self._long_task = self.runner.start(job, done, failed, progress, cancelled)

    def _cancel_long_task(self) -> None:
        if self._long_task is not None:
            self._long_task.cancel()

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt signature
        self.audition.stop()
        self.runner.cancel_all()
        if self._long_task is not None:
            self._long_task.cancel()
        self.session.save()
        super().closeEvent(event)
