"""Top bar: ticker input + date picker + load / backfill buttons."""
from __future__ import annotations

from datetime import date, timedelta

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDateEdit, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget,
)


class TickerBar(QWidget):
    load_requested = pyqtSignal(str, object)         # ticker, date
    backfill_requested = pyqtSignal(str, object, object)  # ticker, start, end

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(QLabel("Ticker:"))
        self.ticker_edit = QLineEdit()
        self.ticker_edit.setFixedWidth(100)
        self.ticker_edit.setPlaceholderText("SPY")
        self.ticker_edit.returnPressed.connect(self._on_load_clicked)
        layout.addWidget(self.ticker_edit)

        layout.addWidget(QLabel("Date:"))
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(self._last_trading_day_qdate())
        layout.addWidget(self.date_edit)

        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)

        self.backfill_btn = QPushButton("Backfill history")
        self.backfill_btn.setToolTip(
            "Download EOD chains for every trading day since 2024-02-05 (can take a while)"
        )
        self.backfill_btn.clicked.connect(self._on_backfill_clicked)
        layout.addWidget(self.backfill_btn)

        layout.addStretch(1)

    @staticmethod
    def _last_trading_day_qdate():
        from PyQt5.QtCore import QDate
        d = date.today() - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return QDate(d.year, d.month, d.day)

    def current_ticker(self) -> str:
        return self.ticker_edit.text().strip().upper() or "SPY"

    def current_date(self) -> date:
        qd = self.date_edit.date()
        return date(qd.year(), qd.month(), qd.day())

    def _on_load_clicked(self):
        self.load_requested.emit(self.current_ticker(), self.current_date())

    def _on_backfill_clicked(self):
        start = date(2024, 2, 5)
        end = self.current_date()
        self.backfill_requested.emit(self.current_ticker(), start, end)
