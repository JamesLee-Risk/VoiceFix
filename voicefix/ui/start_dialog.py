"""First screen: create a project (pick a recording and a flow) or open one."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from voicefix.core.audio_io import SUPPORTED_INPUT_SUFFIXES
from voicefix.core.multitrack import MAX_TRACKS, is_multitrack_file
from voicefix.core.project import FLOW_LABELS, PROJECT_SUFFIX, Flow
from voicefix.steps.registry import IMPLEMENTED_FLOWS
from voicefix.ui.multisession import MultiSession
from voicefix.ui.session import Session, default_cache_root
from voicefix.ui.workers import TaskRunner

AUDIO_FILTER = "錄音檔 (" + " ".join(f"*{s}" for s in SUPPORTED_INPUT_SUFFIXES) + ")"
PROJECT_FILTER = f"VoiceFix 專案 (*{PROJECT_SUFFIX})"
FLOW_HINTS = {
    Flow.VOICE_WORK: "ASMR、廣播劇、雙耳錄音。重點是低底噪、保留表演音、音量一致。",
    Flow.SINGING: "翻唱、原創歌曲的主唱。可以放入伴奏，在伴奏中試聽效果。",
    Flow.MULTITRACK: f"1 條主唱＋最多 {MAX_TRACKS - 1} 條和聲／疊唱（請從同一個錄音專案匯出，起點對齊）。和聲會自動對齊主唱。",
}


class StartDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VoiceFix")
        self.setMinimumWidth(460)
        self.session: Optional[Session] = None
        self.runner = TaskRunner(self)

        title = QLabel("把錄音一步一步處理乾淨", styleSheet="font-size: 20px; font-weight: bold;")
        self.flow_group = QButtonGroup(self)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(QLabel("1. 選擇流程"))
        for i, flow in enumerate(Flow):
            button = QRadioButton(FLOW_LABELS[flow])
            button.setEnabled(flow in IMPLEMENTED_FLOWS)
            button.setChecked(flow == Flow.VOICE_WORK)
            self.flow_group.addButton(button, i)
            hint = QLabel(FLOW_HINTS[flow], wordWrap=True, styleSheet="color: gray; margin-left: 22px;")
            layout.addWidget(button)
            layout.addWidget(hint)
        new_button = QPushButton("2. 選擇錄音並建立專案…")
        open_button = QPushButton("開啟既有專案…")
        layout.addSpacing(8)
        layout.addWidget(new_button)
        layout.addWidget(open_button)
        new_button.clicked.connect(self._new_project)
        open_button.clicked.connect(self._open_project)

    def _selected_flow(self) -> Flow:
        return list(Flow)[self.flow_group.checkedId()]

    def _new_project(self) -> None:
        if self._selected_flow() == Flow.MULTITRACK:
            self._new_multitrack()
            return
        audio, _ = QFileDialog.getOpenFileName(self, "選擇錄音", "", AUDIO_FILTER)
        if not audio:
            return
        source = Path(audio)
        suggested = str(source.with_suffix(PROJECT_SUFFIX))
        target, _ = QFileDialog.getSaveFileName(self, "專案存放位置", suggested, PROJECT_FILTER)
        if not target:
            return
        flow, project_path = self._selected_flow(), Path(target).with_suffix(PROJECT_SUFFIX)
        self._load(lambda progress: Session.create(flow, source, project_path, default_cache_root(), progress))

    def _new_multitrack(self) -> None:
        lead, _ = QFileDialog.getOpenFileName(self, "選擇主唱", "", AUDIO_FILTER)
        if not lead:
            return
        others, _ = QFileDialog.getOpenFileNames(
            self, f"選擇和聲／疊唱（可多選，最多 {MAX_TRACKS - 1} 條）", str(Path(lead).parent), AUDIO_FILTER
        )
        others = [o for o in others if Path(o) != Path(lead)]
        if len(others) > MAX_TRACKS - 1:
            QMessageBox.warning(self, "VoiceFix", f"最多 {MAX_TRACKS - 1} 條和聲／疊唱，已只取前 {MAX_TRACKS - 1} 條。")
            others = others[: MAX_TRACKS - 1]
        sources = [Path(lead), *(Path(o) for o in others)]
        suggested = str(sources[0].with_suffix(PROJECT_SUFFIX))
        target, _ = QFileDialog.getSaveFileName(self, "專案存放位置", suggested, PROJECT_FILTER)
        if not target:
            return
        names = [source.stem for source in sources]
        project_path = Path(target).with_suffix(PROJECT_SUFFIX)
        self._load(lambda progress: MultiSession.create(names, sources, project_path, default_cache_root(), progress))

    def _open_project(self) -> None:
        target, _ = QFileDialog.getOpenFileName(self, "開啟專案", "", PROJECT_FILTER)
        if not target:
            return
        kind = MultiSession if is_multitrack_file(target) else Session
        self._load(lambda progress: kind.open(Path(target), default_cache_root(), progress))

    def _load(self, factory) -> None:
        dialog = QProgressDialog("讀取錄音中…", "", 0, 1000, self)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(300)
        dialog.setAutoClose(False)

        def job(progress, cancel):
            return factory(lambda fraction: progress("讀取錄音中…", fraction))

        def done(session: Session) -> None:
            dialog.close()
            self.session = session
            self.accept()

        def failed(message: str) -> None:
            dialog.close()
            QMessageBox.warning(self, "VoiceFix", message)

        self.runner.start(
            job,
            done,
            failed,
            on_progress=lambda _, fraction: dialog.setValue(int(fraction * 1000)),
        )
