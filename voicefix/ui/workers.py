"""Run heavy work off the UI thread with progress, cancel and stale-result guard."""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from voicefix.core.chunking import Cancelled

log = logging.getLogger(__name__)

Job = Callable[[Callable[[str, float], None], threading.Event], Any]


class _Signals(QObject):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class Task(QRunnable):
    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job
        self.cancel_event = threading.Event()
        self.signals = _Signals()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            result = self.job(self.signals.progress.emit, self.cancel_event)
        except Cancelled:
            self.signals.cancelled.emit()
        except Exception as exc:  # report every failure to the UI instead of dying silently
            log.error("Background task failed:\n%s", traceback.format_exc())
            self.signals.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            if self.cancel_event.is_set():
                self.signals.cancelled.emit()
            else:
                self.signals.done.emit(result)


class TaskRunner(QObject):
    """Starts tasks; ``latest`` slots drop results from superseded runs."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self._latest: dict[str, Task] = {}
        self._running: set[Task] = set()

    def start(
        self,
        job: Job,
        on_done: Callable[[Any], None],
        on_failed: Callable[[str], None],
        on_progress: Callable[[str, float], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        slot: str | None = None,
    ) -> Task:
        task = Task(job)
        task.setAutoDelete(False)
        self._running.add(task)  # keep the signal object alive until delivery
        for signal in (task.signals.done, task.signals.failed, task.signals.cancelled):
            signal.connect(lambda *_: self._running.discard(task))
        if slot is not None:
            previous = self._latest.get(slot)
            if previous is not None:
                previous.cancel()
            self._latest[slot] = task

        def is_current() -> bool:
            return slot is None or self._latest.get(slot) is task

        task.signals.done.connect(lambda result: on_done(result) if is_current() else None)
        task.signals.failed.connect(lambda msg: on_failed(msg) if is_current() else None)
        if on_progress:
            task.signals.progress.connect(on_progress)
        if on_cancelled:
            task.signals.cancelled.connect(on_cancelled)
        self.pool.start(task)
        return task

    def cancel_all(self) -> None:
        for task in self._latest.values():
            task.cancel()
        self._latest.clear()
