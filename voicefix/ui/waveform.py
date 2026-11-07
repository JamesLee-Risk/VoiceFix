"""Waveform (before / after overlay) and spectrogram with region editing.

Left-drag selects (Ctrl adds another selection), click seeks, wheel zooms,
right-drag pans. All positions on the public API are in samples.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from scipy import signal

from voicefix.core.regions import Region
from voicefix.ui.peaks import PeakPyramid

BEFORE_COLOR = (150, 150, 160, 150)
AFTER_COLOR = (52, 152, 219, 200)
SELECTION_BRUSH = (52, 152, 219, 45)
PROTECTED_BRUSH = (46, 204, 113, 55)
MARK_BRUSH = (243, 156, 18, 35)
EVENT_BRUSH = (231, 76, 60, 90)
EVENT_OFF_BRUSH = (149, 165, 166, 60)
PREVIEW_BRUSH = (255, 255, 255, 18)
SPECTROGRAM_MAX_SECONDS = 30.0
REFRESH_MS = 30


class _TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # noqa: ANN001 - pyqtgraph signature
        return [f"{int(v // 60):02d}:{v % 60:04.1f}" if v >= 0 else "" for v in values]


class _NoteAxis(pg.AxisItem):
    """MIDI note numbers shown as note names (C4, D4 ...)."""

    def tickStrings(self, values, scale, spacing):  # noqa: ANN001 - pyqtgraph signature
        names = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
        return [f"{names[int(round(v)) % 12]}{int(round(v)) // 12 - 1}" for v in values]

    def tickValues(self, minVal, maxVal, size):  # noqa: ANN001, N803
        return [(1.0, [float(v) for v in range(int(minVal), int(maxVal) + 1)])]


class _SelectViewBox(pg.ViewBox):
    select_finished = Signal(float, float, bool)  # seconds, additive
    seek = Signal(float)

    def __init__(self) -> None:
        super().__init__(enableMenu=False)
        self.setMouseEnabled(x=True, y=False)
        self._rubber = pg.LinearRegionItem(brush=pg.mkBrush(*SELECTION_BRUSH), movable=False)
        self._rubber.setZValue(100)
        self._rubber.hide()
        self.addItem(self._rubber, ignoreBounds=True)

    def mouseClickEvent(self, ev) -> None:  # noqa: ANN001
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self.seek.emit(self.mapSceneToView(ev.scenePos()).x())
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None) -> None:  # noqa: ANN001
        ev.accept()
        if ev.button() == Qt.MouseButton.LeftButton:
            x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
            x1 = self.mapSceneToView(ev.scenePos()).x()
            self._rubber.setRegion((min(x0, x1), max(x0, x1)))
            self._rubber.show()
            if ev.isFinish():
                self._rubber.hide()
                additive = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.select_finished.emit(min(x0, x1), max(x0, x1), additive)
        elif ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            last = self.mapSceneToView(ev.lastScenePos()).x()
            now = self.mapSceneToView(ev.scenePos()).x()
            self.translateBy(x=last - now)


NOTE_AUTO_COLOR = (46, 204, 113)
NOTE_MANUAL_COLOR = (231, 126, 34)
NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def _note_label(cents: float) -> str:
    midi = int(round(cents / 100.0))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


class WaveformView(QWidget):
    selections_changed = Signal()
    seek_requested = Signal(int)
    pitch_clicked = Signal(int)  # sample position clicked on the pitch plot

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sr = 48_000
        self._pyramid: Optional[PeakPyramid] = None
        self._after: Optional[PeakPyramid] = None
        self._after_offset = 0
        self._selections: list[pg.LinearRegionItem] = []
        self._decor: dict[str, list[pg.GraphicsObject]] = {"protected": [], "marks": [], "events": [], "preview": []}
        self._spectro_source: Optional[tuple[np.ndarray, int]] = None

        self.view_box = _SelectViewBox()
        self.plot = pg.PlotWidget(viewBox=self.view_box, axisItems={"bottom": _TimeAxis("bottom")})
        self.plot.setYRange(-1.0, 1.0)
        self.plot.hideAxis("left")
        self.plot.setMenuEnabled(False)
        self.before_curves = self._make_fill(BEFORE_COLOR)
        self.after_curves = self._make_fill(AFTER_COLOR)
        self.playhead = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen((255, 80, 80), width=1.5))
        self.plot.addItem(self.playhead, ignoreBounds=True)

        self.spectro_plot = pg.PlotWidget(axisItems={"bottom": _TimeAxis("bottom")})
        self.spectro_plot.setXLink(self.plot)
        self.spectro_plot.setMouseEnabled(x=False, y=False)
        self.spectro_plot.setLabel("left", "Hz")
        self.spectro_image = pg.ImageItem()
        self.spectro_image.setColorMap(pg.colormap.get("inferno"))
        self.spectro_plot.addItem(self.spectro_image)
        self.spectro_plot.hide()

        self.pitch_plot = pg.PlotWidget(axisItems={"bottom": _TimeAxis("bottom"), "left": _NoteAxis("left")})
        self.pitch_plot.setXLink(self.plot)
        self.pitch_plot.setMouseEnabled(x=False, y=False)
        self.pitch_plot.showGrid(y=True, alpha=0.25)
        self.pitch_curve = pg.PlotDataItem(pen=pg.mkPen((243, 156, 18), width=1.5), connect="finite")
        self.pitch_plot.addItem(self.pitch_curve)
        self.target_lines = pg.PlotDataItem(pen=pg.mkPen(NOTE_AUTO_COLOR, width=2, style=Qt.PenStyle.DashLine), connect="pairs")
        self.manual_lines = pg.PlotDataItem(pen=pg.mkPen(NOTE_MANUAL_COLOR, width=2.5), connect="pairs")
        self.pitch_plot.addItem(self.target_lines)
        self.pitch_plot.addItem(self.manual_lines)
        self.selected_note = pg.LinearRegionItem(brush=pg.mkBrush(52, 152, 219, 60), movable=False)
        self.selected_note.hide()
        self.pitch_plot.addItem(self.selected_note, ignoreBounds=True)
        self._note_labels: list[pg.TextItem] = []
        self.pitch_plot.scene().sigMouseClicked.connect(self._on_pitch_click)
        self.pitch_plot.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 3)
        layout.addWidget(self.pitch_plot, 2)
        layout.addWidget(self.spectro_plot, 2)

        self._refresh_timer = QTimer(self, singleShot=True, interval=REFRESH_MS, timeout=self._redraw)
        self.view_box.sigXRangeChanged.connect(lambda *_: self._refresh_timer.start())
        self.view_box.select_finished.connect(self._on_select)
        self.view_box.seek.connect(lambda x: self.seek_requested.emit(int(max(0.0, x) * self.sr)))

    def _make_fill(self, color: tuple[int, ...]) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
        upper = pg.PlotDataItem(pen=pg.mkPen(color[:3]))
        lower = pg.PlotDataItem(pen=pg.mkPen(color[:3]))
        fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(*color))
        for item in (upper, lower, fill):
            self.plot.addItem(item)
        return upper, lower

    # ----- content ----------------------------------------------------------
    @property
    def has_input(self) -> bool:
        return self._pyramid is not None

    def set_input(self, audio: np.ndarray, sr: int, reset_view: bool = True) -> None:
        self.sr = sr
        self._pyramid = PeakPyramid(audio)
        self._after = None
        self.view_box.setLimits(xMin=0, xMax=max(len(audio) / sr, 0.1), minXRange=0.01)
        if reset_view:
            self.plot.setXRange(0, max(len(audio) / sr, 0.1), padding=0)
        self._redraw()

    def set_after(self, audio: Optional[np.ndarray], offset: int) -> None:
        self._after = PeakPyramid(audio) if audio is not None and len(audio) else None
        self._after_offset = offset
        self._redraw()

    def _redraw(self) -> None:
        if self._pyramid is None:
            return
        x0, x1 = self.view_box.viewRange()[0]
        width = max(200, self.plot.width())
        start, end = int(max(0, x0) * self.sr), int(x1 * self.sr) + 1
        self._draw(self.before_curves, self._pyramid, start, end, width, 0)
        if self._after is not None:
            lo, hi = start - self._after_offset, end - self._after_offset
            self._draw(self.after_curves, self._after, lo, hi, width, self._after_offset)
        else:
            for curve in self.after_curves:
                curve.setData([], [])

    def _draw(self, curves, pyramid: PeakPyramid, start: int, end: int, width: int, offset: int) -> None:
        x, lower, upper = pyramid.envelope(start, end, width)
        seconds = (x + offset) / self.sr
        curves[0].setData(seconds, upper)
        curves[1].setData(seconds, lower)

    # ----- spectrogram ------------------------------------------------------
    def set_spectrogram_visible(self, visible: bool) -> None:
        self.spectro_plot.setVisible(visible)
        if visible:
            self._draw_spectrogram()

    def set_spectrogram_source(self, audio: Optional[np.ndarray], offset: int) -> None:
        self._spectro_source = (audio, offset) if audio is not None else None
        if self.spectro_plot.isVisible():
            self._draw_spectrogram()

    def _draw_spectrogram(self) -> None:
        if self._spectro_source is None:
            self.spectro_image.clear()
            return
        audio, offset = self._spectro_source
        audio = audio[: int(SPECTROGRAM_MAX_SECONDS * self.sr)]
        if len(audio) < 1024:
            self.spectro_image.clear()
            return
        freqs, _, spec = signal.spectrogram(audio.mean(axis=1), fs=self.sr, nperseg=1024, noverlap=768)
        image = 10 * np.log10(spec + 1e-12)
        self.spectro_image.setImage(image.T, autoLevels=False, levels=(-120, -20))
        self.spectro_image.setRect(QRectF(offset / self.sr, 0, len(audio) / self.sr, freqs[-1]))
        self.spectro_plot.setYRange(0, min(freqs[-1], 16_000))

    # ----- pitch --------------------------------------------------------------
    def set_pitch(self, cents: Optional[np.ndarray], frame: int) -> None:
        """Pitch curve (cents, NaN = unvoiced) on 10 ms frames; None hides the plot."""
        if cents is None or not np.any(~np.isnan(cents)):
            self.pitch_curve.setData([], [])
            self.pitch_plot.hide()
            return
        seconds = (np.arange(len(cents)) * frame + frame / 2.0) / self.sr
        notes = cents / 100.0
        self.pitch_curve.setData(seconds, notes)
        lo, hi = np.nanpercentile(notes, 2), np.nanpercentile(notes, 98)
        self.pitch_plot.setYRange(np.floor(lo) - 1, np.ceil(hi) + 1, padding=0)
        self.pitch_plot.show()

    def set_pitch_notes(self, notes: Sequence[dict], selected: Optional[Region]) -> None:
        """Each note's target as a short line with its name; hand-picked ones in orange."""
        for label in self._note_labels:
            self.pitch_plot.removeItem(label)
        self._note_labels = []
        auto_x, auto_y, manual_x, manual_y = [], [], [], []
        for note in notes:
            x0, x1 = note["region"].start / self.sr, note["region"].end / self.sr
            y = note["target"] / 100.0
            xs, ys = (manual_x, manual_y) if note["manual"] else (auto_x, auto_y)
            xs += [x0, x1]
            ys += [y, y]
            color = NOTE_MANUAL_COLOR if note["manual"] else NOTE_AUTO_COLOR
            label = pg.TextItem(_note_label(note["target"]) + ("（手動）" if note["manual"] else ""), color=color, anchor=(0, 1))
            label.setPos(x0, y + 0.3)
            self.pitch_plot.addItem(label, ignoreBounds=True)
            self._note_labels.append(label)
        self.target_lines.setData(auto_x, auto_y)
        self.manual_lines.setData(manual_x, manual_y)
        if selected is None:
            self.selected_note.hide()
        else:
            self.selected_note.setRegion((selected.start / self.sr, selected.end / self.sr))
            self.selected_note.show()

    def _on_pitch_click(self, ev) -> None:  # noqa: ANN001 - pyqtgraph signature
        view = self.pitch_plot.plotItem.vb
        if ev.button() != Qt.MouseButton.LeftButton or not view.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.pitch_clicked.emit(int(max(0.0, view.mapSceneToView(ev.scenePos()).x()) * self.sr))

    # ----- regions ----------------------------------------------------------
    def _on_select(self, x0: float, x1: float, additive: bool) -> None:
        if x1 - x0 < 0.01:
            return
        if not additive:
            self.clear_selections(emit=False)
        self._add_selection(x0, x1)
        self.selections_changed.emit()

    def _add_selection(self, x0: float, x1: float) -> None:
        item = pg.LinearRegionItem((x0, x1), brush=pg.mkBrush(*SELECTION_BRUSH))
        item.setZValue(50)
        item.sigRegionChangeFinished.connect(lambda *_: self.selections_changed.emit())
        self.plot.addItem(item)
        self._selections.append(item)

    def selections(self) -> list[Region]:
        out = []
        for item in self._selections:
            lo, hi = item.getRegion()
            out.append(Region(int(max(0.0, lo) * self.sr), int(max(0.0, hi) * self.sr)))
        return sorted(r for r in out if r.length > 0)

    def set_selections(self, regions: Sequence[Region]) -> None:
        self.clear_selections(emit=False)
        for r in regions:
            self._add_selection(r.start / self.sr, r.end / self.sr)

    def clear_selections(self, emit: bool = True) -> None:
        for item in self._selections:
            self.plot.removeItem(item)
        self._selections.clear()
        if emit:
            self.selections_changed.emit()

    def _set_decor(self, kind: str, regions: Sequence[Region], brushes: Sequence[tuple[int, ...]], z: float) -> None:
        for item in self._decor[kind]:
            self.plot.removeItem(item)
        self._decor[kind] = []
        for region, brush in zip(regions, brushes):
            item = pg.LinearRegionItem((region.start / self.sr, region.end / self.sr), brush=pg.mkBrush(*brush), movable=False)
            item.setZValue(z)
            for line in item.lines:
                line.setPen(pg.mkPen(None))
            self.plot.addItem(item, ignoreBounds=True)
            self._decor[kind].append(item)

    def set_protected(self, regions: Sequence[Region]) -> None:
        self._set_decor("protected", regions, [PROTECTED_BRUSH] * len(regions), 20)

    def set_marks(self, regions: Sequence[Region]) -> None:
        self._set_decor("marks", regions, [MARK_BRUSH] * len(regions), 10)

    def set_events(self, regions: Sequence[Region], enabled: Sequence[bool]) -> None:
        brushes = [EVENT_BRUSH if on else EVENT_OFF_BRUSH for on in enabled]
        widened = [Region(r.start, max(r.end, r.start + int(self.sr * 0.004))) for r in regions]  # keep tiny clicks visible
        self._set_decor("events", widened, brushes, 30)

    def set_preview_span(self, span: Optional[Region]) -> None:
        self._set_decor("preview", [span] if span else [], [PREVIEW_BRUSH], 5)

    def set_playhead(self, sample: int) -> None:
        self.playhead.setValue(sample / self.sr)

    def zoom_to(self, region: Region, margin_seconds: float = 0.5) -> None:
        self.plot.setXRange(max(0.0, region.start / self.sr - margin_seconds), region.end / self.sr + margin_seconds, padding=0)
