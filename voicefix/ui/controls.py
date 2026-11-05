"""The right-hand column: one strength slider, at most one option, scope,
event list and a collapsed advanced section."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from voicefix.core.project import StepConfig
from voicefix.core.regions import SCOPE_LABELS, ScopeMode
from voicefix.core.timefmt import format_time
from voicefix.steps.base import Event, Step

WARNING_STYLE = "color: #e67e22; font-weight: bold;"


class StepControls(QWidget):
    """Edits a draft StepConfig; emits ``changed`` whenever it differs."""

    changed = Signal()
    reset_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._step: Optional[Step] = None
        self._draft = StepConfig()
        self._loading = False

        self.strength_box = QGroupBox("強度")
        self.slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=100, pageStep=10)
        self.value_label = QLabel("0")
        self.value_label.setMinimumWidth(28)
        self.warning = QLabel(wordWrap=True, styleSheet=WARNING_STYLE)
        row = QHBoxLayout()
        row.addWidget(self.slider)
        row.addWidget(self.value_label)
        box_layout = QVBoxLayout(self.strength_box)
        box_layout.addLayout(row)
        box_layout.addWidget(self.warning)

        self.option_box = QGroupBox()
        self.option_combo = QComboBox()
        QVBoxLayout(self.option_box).addWidget(self.option_combo)

        self.scope_box = QGroupBox("套用範圍")
        self.scope_group = QButtonGroup(self)
        scope_layout = QVBoxLayout(self.scope_box)
        for i, mode in enumerate((ScopeMode.WHOLE, ScopeMode.SELECTION, ScopeMode.EXCLUDE_SELECTION)):
            button = QRadioButton(SCOPE_LABELS[mode])
            button.setProperty("scope", mode.value)
            self.scope_group.addButton(button, i)
            scope_layout.addWidget(button)

        self.advanced_box = QGroupBox("進階（真實參數）", checkable=True, checked=False)
        self.advanced_body = QWidget()
        self.advanced_form = QFormLayout(self.advanced_body)
        QVBoxLayout(self.advanced_box).addWidget(self.advanced_body)
        self.advanced_body.setVisible(False)
        self.advanced_box.toggled.connect(self.advanced_body.setVisible)
        self._spins: dict[str, QDoubleSpinBox] = {}

        self.reset_button = QPushButton("重設為建議值")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.strength_box, self.option_box, self.scope_box, self.advanced_box, self.reset_button):
            layout.addWidget(widget)

        self.slider.valueChanged.connect(self._on_strength)
        self.option_combo.currentIndexChanged.connect(self._on_option)
        self.scope_group.idToggled.connect(self._on_scope)
        self.reset_button.clicked.connect(self.reset_requested)

    # ----- loading ----------------------------------------------------------
    def load(self, step: Step, draft: StepConfig) -> None:
        self._loading = True
        self._step, self._draft = step, draft
        self.strength_box.setVisible(step.has_strength)
        self.strength_box.setTitle(step.strength_label)
        self.slider.setValue(draft.strength if draft.strength is not None else step.default_strength)
        self.value_label.setText(str(self.slider.value()))

        self.option_box.setVisible(bool(step.options))
        self.option_box.setTitle(step.option_label)
        self.option_combo.clear()
        for option in step.options:
            self.option_combo.addItem(option.label, option.key)
        keys = [o.key for o in step.options]
        current = draft.option or step.default_option
        if current in keys:
            self.option_combo.setCurrentIndex(keys.index(current))

        general = [m for m in step.scopes if m != ScopeMode.EVENTS]
        self.scope_box.setVisible(len(general) > 1)
        for button in self.scope_group.buttons():
            button.setChecked(button.property("scope") == draft.scope.value)

        self._build_advanced(step)
        self.advanced_box.setVisible(bool(step.advanced))
        self.reset_button.setVisible(step.has_strength or bool(step.options) or bool(step.advanced))
        self._loading = False
        self._refresh_derived()

    def _build_advanced(self, step: Step) -> None:
        while self.advanced_form.rowCount():
            self.advanced_form.removeRow(0)
        self._spins = {}
        for spec in step.advanced:
            spin = QDoubleSpinBox(decimals=spec.decimals, minimum=spec.minimum, maximum=spec.maximum)
            spin.setSuffix(f" {spec.unit}" if spec.unit else "")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda value, key=spec.key: self._on_advanced(key, value))
            self.advanced_form.addRow(spec.label, spin)
            self._spins[spec.key] = spin

    @property
    def draft(self) -> StepConfig:
        return self._draft

    # ----- edits ------------------------------------------------------------
    def _emit(self, draft: StepConfig) -> None:
        self._draft = draft
        self._refresh_derived()
        if not self._loading:
            self.changed.emit()

    def _on_strength(self, value: int) -> None:
        self.value_label.setText(str(value))
        if not self._loading:
            self._emit(replace(self._draft, strength=value))

    def _on_option(self, index: int) -> None:
        if not self._loading and index >= 0:
            self._emit(replace(self._draft, option=self.option_combo.itemData(index)))

    def _on_scope(self, button_id: int, checked: bool) -> None:
        if checked and not self._loading:
            mode = ScopeMode(self.scope_group.button(button_id).property("scope"))
            self._emit(replace(self._draft, scope=mode))

    def _on_advanced(self, key: str, value: float) -> None:
        if not self._loading:
            self._emit(replace(self._draft, advanced={**self._draft.advanced, key: value}))

    def _refresh_derived(self) -> None:
        """Warning text and the real parameter values behind the slider."""
        step = self._step
        if step is None:
            return
        strength = self._draft.strength if self._draft.strength is not None else step.default_strength
        warning = step.warning(strength, self._draft.option or step.default_option)
        self.warning.setText(f"⚠ {warning}" if warning else "")
        self.warning.setVisible(bool(warning))
        params = step.resolve_params(self._draft.strength, self._draft.option, self._draft.advanced)
        was_loading, self._loading = self._loading, True
        for key, spin in self._spins.items():
            if key in params:
                spin.setValue(params[key])
            custom = key in self._draft.advanced
            label = self.advanced_form.labelForField(spin)
            if label is not None:
                spec = next(s for s in step.advanced if s.key == key)
                label.setText(spec.label + ("（已自訂）" if custom else ""))
        self._loading = was_loading


class EventList(QWidget):
    """Detected events: tick to process, click to jump there."""

    toggled = Signal()
    activated = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = QLabel()
        self.list = QListWidget()
        self.all_on = QPushButton("全部勾選")
        self.all_off = QPushButton("全部取消")
        buttons = QHBoxLayout()
        buttons.addWidget(self.all_on)
        buttons.addWidget(self.all_off)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)
        self._events: list[Event] = []
        self._loading = False
        self.list.itemChanged.connect(lambda _: None if self._loading else self.toggled.emit())
        self.list.currentRowChanged.connect(lambda row: self.activated.emit(row) if row >= 0 else None)
        self.all_on.clicked.connect(lambda: self._set_all(True))
        self.all_off.clicked.connect(lambda: self._set_all(False))

    def set_events(self, events: Sequence[Event], enabled: Sequence[bool], sr: int, label: str) -> None:
        self._loading = True
        self._events = list(events)
        self.list.clear()
        for event, on in zip(events, enabled):
            item = QListWidgetItem(f"{event.label}　{format_time(event.region.start, sr)}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.title.setText(f"偵測到 {len(events)} 個{label}（取消勾選＝不處理）")
        self._loading = False

    def set_busy(self, text: str) -> None:
        self.title.setText(text)

    def enabled(self) -> list[bool]:
        return [self.list.item(i).checkState() == Qt.CheckState.Checked for i in range(self.list.count())]

    def events(self) -> list[Event]:
        return list(self._events)

    def _set_all(self, on: bool) -> None:
        self._loading = True
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._loading = False
        self.toggled.emit()
