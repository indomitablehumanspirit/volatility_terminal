"""Reusable panel for adding multiple (ticker, date) comparison snapshots."""
from __future__ import annotations

from datetime import date

from PyQt5.QtCore import QDate, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QDateEdit, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)


class _AddDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add comparison")
        layout = QFormLayout(self)
        self.ticker_edit = QLineEdit()
        self.ticker_edit.setPlaceholderText("e.g. QQQ")
        layout.addRow("Ticker:", self.ticker_edit)
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate().addDays(-1))
        layout.addRow("Date:", self.date_edit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def values(self) -> tuple[str, date]:
        t = self.ticker_edit.text().strip().upper()
        qd = self.date_edit.date()
        return t, date(qd.year(), qd.month(), qd.day())


class ComparisonPanel(QWidget):
    """Sidebar widget managing a list of (ticker, date) comparison entries.

    Emits load_requested(entry_id, ticker, day) when user adds an entry.
    Emits entry_removed(entry_id) when user removes one.
    entry_id values are stable (monotonically increasing, never reused).
    """

    load_requested = pyqtSignal(int, str, object)  # entry_id, ticker, date
    entry_removed = pyqtSignal(int)                # entry_id

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        layout.addWidget(QLabel("Compare:"))

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(110)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+ Add")
        self.remove_btn = QPushButton("Remove")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)

        self.add_btn.clicked.connect(self._on_add)
        self.remove_btn.clicked.connect(self._on_remove)

        self._next_id = 0
        self._entries: list[dict] = []  # [{'id', 'ticker', 'day'}]

    def _on_add(self):
        dlg = _AddDialog(self)
        if not dlg.exec_():
            return
        ticker, day = dlg.values()
        if not ticker:
            return
        entry_id = self._next_id
        self._next_id += 1
        self._entries.append({"id": entry_id, "ticker": ticker, "day": day})
        item = QListWidgetItem(f"{ticker}  {day}  ⧗")
        item.setData(Qt.UserRole, entry_id)
        self.list_widget.addItem(item)
        self.load_requested.emit(entry_id, ticker, day)

    def _on_remove(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        item = self.list_widget.item(row)
        entry_id = item.data(Qt.UserRole)
        self.list_widget.takeItem(row)
        self._entries = [e for e in self._entries if e["id"] != entry_id]
        self.entry_removed.emit(entry_id)

    def set_entry_status(self, entry_id: int, status: str):
        """Update label for an entry. status: 'ready' or 'failed'."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == entry_id:
                entry = next(
                    (e for e in self._entries if e["id"] == entry_id), None
                )
                if entry is None:
                    return
                suffix = "✓" if status == "ready" else "✗ failed"
                item.setText(f"{entry['ticker']}  {entry['day']}  {suffix}")
                return
