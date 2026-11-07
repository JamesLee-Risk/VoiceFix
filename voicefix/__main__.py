"""Entry point: ``python -m voicefix`` (or ``VoiceFix.exe``).

``--self-test <report.txt>`` checks the installation instead of opening the app.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _log_file() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoiceFix"
    base.mkdir(parents=True, exist_ok=True)
    return base / "voicefix.log"


def _setup_logging() -> None:
    """A packaged (windowed) build has no console, so it logs to a file, and
    uncaught errors end up there too instead of vanishing."""
    if not getattr(sys, "frozen", False):
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
        return
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename=_log_file(), encoding="utf-8")

    def log_uncaught(kind, value, tb):  # noqa: ANN001 - sys.excepthook signature
        logging.getLogger("voicefix").critical("Uncaught error", exc_info=(kind, value, tb))

    sys.excepthook = log_uncaught


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--self-test":
        from voicefix.selftest import run

        return run(Path(sys.argv[2]))
    _setup_logging()
    import pyqtgraph as pg
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from voicefix.ui.main_window import MainWindow
    from voicefix.ui.start_dialog import StartDialog

    app = QApplication(sys.argv)
    app.setApplicationName("VoiceFix")
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft JhengHei UI", 10))
    pg.setConfigOptions(background="#1e1f22", foreground="#d0d0d0", antialias=False)

    start = StartDialog()
    if start.exec() != StartDialog.DialogCode.Accepted or start.session is None:
        return 0
    window = MainWindow(start.session)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
