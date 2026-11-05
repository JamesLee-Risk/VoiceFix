"""Left column: every step with its status; only unlocked steps are clickable."""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from voicefix.core.project import STATUS_LABELS, Project, StepStatus
from voicefix.steps.base import Step

STATUS_COLORS = {
    StepStatus.NOT_STARTED: QColor(140, 140, 140),
    StepStatus.APPLIED: QColor(46, 204, 113),
    StepStatus.SKIPPED: QColor(149, 165, 166),
    StepStatus.STALE: QColor(230, 126, 34),
}
STATUS_ICONS = {StepStatus.NOT_STARTED: "○", StepStatus.APPLIED: "●", StepStatus.SKIPPED: "–", StepStatus.STALE: "↻"}


class StepList(QListWidget):
    step_chosen = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setSpacing(2)
        self.setWordWrap(True)
        self.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._updating = False
        self.currentRowChanged.connect(self._on_row)

    def refresh(self, steps: Sequence[Step], project: Project, current: int) -> None:
        self._updating = True
        self.clear()
        for i, step in enumerate(steps):
            status = project.status(i)
            item = QListWidgetItem(f"{STATUS_ICONS[status]}  {step.code}  {step.title}\n      {STATUS_LABELS[status]}")
            item.setForeground(STATUS_COLORS[status] if i != current else QColor(255, 255, 255))
            if not project.is_unlocked(i):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip("請先完成（套用或略過）前面的步驟")
            self.addItem(item)
        self.setCurrentRow(current)
        self._updating = False

    def _on_row(self, row: int) -> None:
        if not self._updating and row >= 0:
            self.step_chosen.emit(row)
