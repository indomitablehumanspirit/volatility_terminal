"""First-run dialog to collect Alpaca API credentials."""
from __future__ import annotations

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout,
)

ORG = "VolatilityTerminal"
APP = "VolatilityTerminal"
KEY = "alpaca/api_key"
SECRET = "alpaca/api_secret"


def load_creds() -> tuple[str, str] | None:
    s = QSettings(ORG, APP)
    k = s.value(KEY, "", type=str)
    sec = s.value(SECRET, "", type=str)
    return (k, sec) if k and sec else None


def save_creds(key: str, secret: str) -> None:
    s = QSettings(ORG, APP)
    s.setValue(KEY, key)
    s.setValue(SECRET, secret)


class CredsDialog(QDialog):
    def __init__(self, parent=None, existing: tuple[str, str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Alpaca API Credentials")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        info = QLabel(
            "Enter your Alpaca API key and secret.\n"
            "Free-tier keys from https://alpaca.markets are sufficient for "
            "historical options data."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.key_edit = QLineEdit()
        self.secret_edit = QLineEdit()
        self.secret_edit.setEchoMode(QLineEdit.Password)
        if existing:
            self.key_edit.setText(existing[0])
            self.secret_edit.setText(existing[1])
        form.addRow("API Key", self.key_edit)
        form.addRow("API Secret", self.secret_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str]:
        return self.key_edit.text().strip(), self.secret_edit.text().strip()


def ensure_creds(parent=None) -> tuple[str, str] | None:
    existing = load_creds()
    if existing:
        return existing
    dlg = CredsDialog(parent)
    if dlg.exec_() == QDialog.Accepted:
        k, s = dlg.values()
        if k and s:
            save_creds(k, s)
            return (k, s)
    return None
