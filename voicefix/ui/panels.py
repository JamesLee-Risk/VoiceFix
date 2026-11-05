"""Extra controls for steps that are not a plain strength slider
(A0/B0 import, A1 editing, A11/B12 export)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from voicefix.core.audio_io import EXPORT_FORMATS
from voicefix.core.regions import Region
from voicefix.core.timefmt import describe_region
from voicefix.dsp.key import KEYS

AUTO_KEY = "auto"

EDIT_KINDS = {"deletions": "刪除", "mutes": "靜音", "ng": "NG"}


class ImportPanel(QWidget):
    binaural_changed = Signal(bool)
    use_selection_as_noise = Signal()
    show_noise = Signal()
    choose_accompaniment = Signal()
    clear_accompaniment = Signal()
    key_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.binaural = QCheckBox("這是雙耳錄音")
        self.binaural.setToolTip("勾選後所有處理左右聲道連動，不會造成左右時間差，也不會轉成單聲道。")
        self.noise_label = QLabel(wordWrap=True)
        self.use_selection = QPushButton("把選取的區段設為底噪樣本")
        self.listen = QPushButton("跳到底噪樣本")
        hint = QLabel("底噪樣本要選一段「沒有說話、只有背景聲」的地方，3 秒左右最好。", wordWrap=True)
        hint.setStyleSheet("color: gray;")

        self.song_box = QWidget()
        self.accompaniment_label = QLabel(wordWrap=True)
        self.choose = QPushButton("選擇伴奏…")
        self.clear = QPushButton("移除伴奏")
        self.key = QComboBox()
        self.key.addItem("自動偵測", AUTO_KEY)
        for name in KEYS:
            self.key.addItem(name, name)
        accompaniment_buttons = QHBoxLayout()
        accompaniment_buttons.addWidget(self.choose)
        accompaniment_buttons.addWidget(self.clear)
        song_layout = QVBoxLayout(self.song_box)
        song_layout.setContentsMargins(0, 0, 0, 8)
        song_layout.addWidget(QLabel("伴奏（只用來試聽與混合試聽版，不會被處理）", wordWrap=True))
        song_layout.addWidget(self.accompaniment_label)
        song_layout.addLayout(accompaniment_buttons)
        song_layout.addWidget(QLabel("歌曲調性"))
        song_layout.addWidget(self.key)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.song_box, self.binaural, self.noise_label, self.use_selection, self.listen, hint):
            layout.addWidget(widget)
        self.binaural.toggled.connect(self.binaural_changed)
        self.use_selection.clicked.connect(self.use_selection_as_noise)
        self.listen.clicked.connect(self.show_noise)
        self.choose.clicked.connect(self.choose_accompaniment)
        self.clear.clicked.connect(self.clear_accompaniment)
        self.key.activated.connect(lambda i: self.key_changed.emit(self.key.itemData(i)))

    def configure(self, singing: bool) -> None:
        """Flow B shows accompaniment and key; binaural is a flow A concept."""
        self.song_box.setVisible(singing)
        self.binaural.setVisible(not singing)

    def set_song_state(self, accompaniment: Optional[str], error: Optional[str], key: str, detected: Optional[str]) -> None:
        if error:
            self.accompaniment_label.setText(f"⚠ {error}")
        else:
            self.accompaniment_label.setText(Path(accompaniment).name if accompaniment else "尚未放入伴奏")
        self.clear.setEnabled(bool(accompaniment))
        self.key.setItemText(0, f"自動偵測（{detected}）" if detected else "自動偵測")
        index = self.key.findData(key or AUTO_KEY)
        self.key.setCurrentIndex(max(index, 0))

    def set_state(self, binaural: bool, noise: Optional[Region], sr: int) -> None:
        self.binaural.blockSignals(True)
        self.binaural.setChecked(binaural)
        self.binaural.blockSignals(False)
        self.noise_label.setText(f"底噪樣本：{describe_region(noise, sr)}" if noise else "尚未設定底噪樣本")
        self.listen.setEnabled(noise is not None)


class EditPanel(QWidget):
    """A1: act on the current selections; list and undo existing edits."""

    edit_requested = Signal(str)  # kind
    remove_requested = Signal(str, int)  # kind, index

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        buttons = QHBoxLayout()
        for kind in ("deletions", "mutes", "ng"):
            button = QPushButton({"deletions": "刪除選取", "mutes": "靜音選取", "ng": "標記 NG"}[kind])
            button.clicked.connect(lambda _=False, k=kind: self.edit_requested.emit(k))
            buttons.addWidget(button)
        self.list = QListWidget()
        self.remove = QPushButton("復原選取的剪輯")
        hint = QLabel("刪除會讓後面的時間往前移；NG 只是標記，不會改變聲音。", wordWrap=True)
        hint.setStyleSheet("color: gray;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.remove)
        layout.addWidget(hint)
        self.remove.clicked.connect(self._on_remove)

    def set_edits(self, settings: Mapping[str, Any], sr: int) -> None:
        self.list.clear()
        for kind, label in EDIT_KINDS.items():
            for i, raw in enumerate(settings.get(kind, [])):
                item = QListWidgetItem(f"{label}　{describe_region(Region.from_json(raw), sr)}")
                item.setData(256, (kind, i))
                self.list.addItem(item)

    def _on_remove(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            kind, index = item.data(256)
            self.remove_requested.emit(kind, index)


class ExportPanel(QWidget):
    export_requested = Signal(str, bool, bool, bool)  # format key, split, dry, mix

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.format = QComboBox()
        for fmt in EXPORT_FORMATS:
            self.format.addItem(fmt.label, fmt.key)
        self.split = QCheckBox("依台詞段落分成多個檔案")
        self.dry = QCheckBox("另外匯出乾聲（不含空間效果）")
        self.mix = QCheckBox("另外匯出與伴奏混合的試聽版")
        self.export = QPushButton("匯出…")
        self.status = QLabel(wordWrap=True)
        hint = QLabel("匯出時會自動重算標示為「需重算」的步驟。原始錄音不會被覆寫。", wordWrap=True)
        hint.setStyleSheet("color: gray;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (QLabel("格式"), self.format, self.split, self.dry, self.mix, self.export, self.status, hint):
            layout.addWidget(widget)
        self.export.clicked.connect(
            lambda: self.export_requested.emit(
                self.format.currentData(),
                self.split.isChecked(),
                not self.dry.isHidden() and self.dry.isEnabled() and self.dry.isChecked(),
                not self.mix.isHidden() and self.mix.isEnabled() and self.mix.isChecked(),
            )
        )

    def configure(self, singing: bool, has_wet_effects: bool, has_backing: bool) -> None:
        self.split.setVisible(not singing)
        self.dry.setVisible(singing)
        self.dry.setEnabled(has_wet_effects)
        self.dry.setToolTip("" if has_wet_effects else "目前沒有套用空間效果，人聲本身就是乾聲。")
        self.mix.setVisible(singing)
        self.mix.setEnabled(has_backing)
        self.mix.setChecked(has_backing and self.mix.isChecked())
        self.mix.setToolTip("" if has_backing else "請先在 B0 放入伴奏。")
