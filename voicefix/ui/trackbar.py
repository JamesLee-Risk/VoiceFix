"""Flow C header row: which track is on screen, and whether it follows the
shared settings of the current step."""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QWidget


class TrackBar(QWidget):
    track_chosen = Signal(int)
    own_settings_changed = Signal(bool)  # True = this track uses its own values for this step

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.tracks = QComboBox()
        self.tracks.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.tracks.setMinimumWidth(160)
        self.tracks.setToolTip("切換目前要檢查、試聽的軌。按「套用」時所有軌會一起計算。")
        self.own = QCheckBox("此軌單獨設定")
        self.own.setToolTip("不勾選時，這一步的設定全部軌共用；勾選後這一軌可以有自己的設定。")
        self.hint = QLabel(styleSheet="color: gray;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("目前的軌"))
        layout.addWidget(self.tracks)
        layout.addWidget(self.own)
        layout.addWidget(self.hint)
        layout.addStretch(1)
        self._loading = False
        self.tracks.currentIndexChanged.connect(lambda i: None if self._loading or i < 0 else self.track_chosen.emit(i))
        self.own.toggled.connect(lambda on: None if self._loading else self.own_settings_changed.emit(on))

    def refresh(self, names: Sequence[str], current: int, own_settings: bool, step_has_settings: bool) -> None:
        self._loading = True
        self.tracks.clear()
        for i, name in enumerate(names):
            self.tracks.addItem(f"{'★ ' if i == 0 else ''}{name}")
        self.tracks.setCurrentIndex(current)
        self.own.setChecked(own_settings)
        self.own.setVisible(step_has_settings)
        self.hint.setText("（★ 主唱是對齊與音量的基準）" if current != 0 else "")
        self._loading = False
