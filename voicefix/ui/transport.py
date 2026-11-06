"""Bottom bar: audition controls and step navigation."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget

from voicefix.playback.mixer import MODE_LABELS, AuditionMode


class TransportBar(QWidget):
    play_toggled = Signal()
    mode_changed = Signal(object)  # AuditionMode
    loop_changed = Signal(bool)
    gain_match_changed = Signal(bool)
    backing_changed = Signal(bool)
    others_changed = Signal(bool)
    spectrogram_changed = Signal(bool)
    previous_requested = Signal()
    skip_requested = Signal()
    apply_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.play = QPushButton("▶ 播放")
        self.play.setToolTip("空白鍵：播放／停止")
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[AuditionMode, QPushButton] = {}
        for i, mode in enumerate(AuditionMode):
            button = QPushButton(MODE_LABELS[mode], checkable=True)
            button.setToolTip("B：切換處理前／處理後" if mode != AuditionMode.REMOVED else "R：只聽被移除的聲音")
            self.mode_group.addButton(button, i)
            self.mode_buttons[mode] = button
        self.mode_buttons[AuditionMode.AFTER].setChecked(True)
        self.loop = QCheckBox("循環", checked=True)
        self.gain_match = QCheckBox("音量對齊", checked=True)
        self.gain_match.setToolTip("把處理後的音量對齊處理前，避免「比較大聲聽起來比較好」的誤判。")
        self.gain_label = QLabel()
        self.gain_label.setStyleSheet("color: gray;")
        self.spectrogram = QCheckBox("頻譜圖")
        self.backing = QCheckBox("伴奏一起播")
        self.backing.setToolTip("在伴奏中判斷效果。伴奏不會被處理。")
        self.backing.setVisible(False)
        self.others = QCheckBox("其他軌一起播")
        self.others.setToolTip("在其他聲部（主唱或和聲）中判斷效果。")
        self.others.setVisible(False)
        self.previous = QPushButton("◀ 上一步")
        self.skip = QPushButton("略過此步")
        self.apply = QPushButton("套用並前往下一步 ▶")
        self.apply.setDefault(True)
        self.apply.setStyleSheet("font-weight: bold;")

        layout = QHBoxLayout(self)
        layout.addWidget(self.play)
        for button in self.mode_buttons.values():
            layout.addWidget(button)
        for widget in (self.loop, self.gain_match, self.gain_label, self.backing, self.others, self.spectrogram):
            layout.addWidget(widget)
        layout.addStretch(1)
        for widget in (self.previous, self.skip, self.apply):
            layout.addWidget(widget)

        modes = list(AuditionMode)
        self.play.clicked.connect(self.play_toggled)
        self.mode_group.idClicked.connect(lambda i: self.mode_changed.emit(modes[i]))
        self.loop.toggled.connect(self.loop_changed)
        self.gain_match.toggled.connect(self.gain_match_changed)
        self.spectrogram.toggled.connect(self.spectrogram_changed)
        self.backing.toggled.connect(self.backing_changed)
        self.others.toggled.connect(self.others_changed)
        self.previous.clicked.connect(self.previous_requested)
        self.skip.clicked.connect(self.skip_requested)
        self.apply.clicked.connect(self.apply_requested)

    def set_mode(self, mode: AuditionMode) -> None:
        self.mode_buttons[mode].setChecked(True)

    def set_playing(self, playing: bool) -> None:
        self.play.setText("■ 停止" if playing else "▶ 播放")

    def set_backing(self, available: bool, checked: bool) -> None:
        self.backing.setVisible(available)
        self.backing.setChecked(available and checked)

    def set_others(self, available: bool, checked: bool) -> None:
        self.others.setVisible(available)
        self.others.setChecked(available and checked)

    def set_gain_db(self, gain_db: Optional[float]) -> None:
        self.gain_label.setText("" if gain_db is None else f"（處理後 {gain_db:+.1f} dB）")

    def set_navigation(self, has_previous: bool, can_skip: bool, apply_text: Optional[str]) -> None:
        self.previous.setEnabled(has_previous)
        self.skip.setEnabled(can_skip)
        self.apply.setVisible(apply_text is not None)
        if apply_text:
            self.apply.setText(apply_text)

    def set_busy(self, busy: bool) -> None:
        for button in (self.previous, self.skip, self.apply):
            button.setDisabled(busy)
