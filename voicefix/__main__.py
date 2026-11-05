"""Entry point: ``python -m voicefix``."""

from __future__ import annotations

import logging
import sys


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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
