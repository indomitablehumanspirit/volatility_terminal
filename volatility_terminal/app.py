"""QApplication bootstrap."""
from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from .data.alpaca_client import AlpacaCreds
from .ui.creds_dialog import ensure_creds
from .ui.main_window import MainWindow


def run() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Volatility Terminal")
    app.setOrganizationName("VolatilityTerminal")

    creds = ensure_creds()
    if not creds:
        return 1
    win = MainWindow(AlpacaCreds(api_key=creds[0], api_secret=creds[1]))
    win.show()
    return app.exec_()
