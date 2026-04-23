"""Earnings tab: historical implied vs realized move + realized-move distribution.

Supports vol-crush trading decisions: "what has this name typically moved on
earnings, and how does today's implied pricing compare to that history?"
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...analytics.earnings import build_earnings_table
from ...data import cache
from ...data.chain_fetcher import ChainFetcher
from ...data.earnings import get_earnings_dates, refresh_earnings_dates
from ..workers import Worker


_COLS = ["Date", "Time", "Implied (straddle) %", "Implied (jump) %",
        "Realized %", "Realized − Implied %"]


def _ensure_bars_cover(ticker: str, earnings: pd.DataFrame,
                       alpaca) -> pd.DataFrame:
    """Return underlying bars covering all earnings dates, fetching if needed.

    The ChainFetcher only caches bars for dates it has touched. Earnings-move
    history goes further back than options coverage, so we top up the cache
    from Alpaca's equity endpoint on demand.
    """
    existing = cache.read_underlying(ticker)
    if earnings is None or earnings.empty:
        return existing if existing is not None else pd.DataFrame()

    need_start = min(earnings["date"]) - timedelta(days=7)
    need_end = max(earnings["date"]) + timedelta(days=7)

    if existing is not None and not existing.empty:
        have = pd.to_datetime(existing["timestamp"]).dt.tz_localize(None).dt.date
        have_start, have_end = have.min(), have.max()
        if have_start <= need_start and have_end >= need_end:
            return existing

    fetched = alpaca.get_daily_stock_bars(ticker, need_start, need_end)
    if fetched is None or fetched.empty:
        return existing if existing is not None else pd.DataFrame()

    if existing is not None and not existing.empty:
        merged = pd.concat([existing, fetched], ignore_index=True)
        merged["_d"] = pd.to_datetime(merged["timestamp"]).dt.tz_localize(None).dt.date
        merged = merged.drop_duplicates("_d").drop(columns="_d")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
    else:
        merged = fetched.sort_values("timestamp").reset_index(drop=True)

    cache.write_underlying(ticker, merged)
    return merged


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_signed_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.2f}%"


class EarningsTab(QWidget):
    def __init__(self, fetcher: ChainFetcher, parent=None):
        super().__init__(parent)
        self._fetcher = fetcher
        self._ticker: Optional[str] = None
        self._pool = QThreadPool.globalInstance()

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        self.status = QLabel("Load a ticker to view earnings history.")
        self.status.setStyleSheet("color: #bbb;")
        header.addWidget(self.status, 1)
        self.refresh_btn = QPushButton("Refresh from yfinance")
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.refresh_btn.setEnabled(False)
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(8)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        body.addWidget(self.table, 3)

        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setBackground("#111")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hist_plot.setLabel("left", "Count")
        self.hist_plot.setLabel("bottom", "Realized move (%)")
        self.hist_plot.setTitle("Realized earnings-move distribution")
        self._hist_bar: Optional[pg.BarGraphItem] = None
        self._hist_lines: list = []
        body.addWidget(self.hist_plot, 2)

        root.addLayout(body, 1)

    # ---- public API -------------------------------------------------------

    def set_ticker(self, ticker: str):
        ticker = ticker.upper().strip()
        if not ticker:
            return
        self._ticker = ticker
        self.refresh_btn.setEnabled(True)
        self.status.setText(f"Loading earnings history for {ticker}…")
        self._load(use_earnings_cache=True)

    # ---- handlers ---------------------------------------------------------

    def _on_refresh(self):
        if not self._ticker:
            return
        self.status.setText(f"Refreshing earnings calendar for {self._ticker}…")
        self._load(use_earnings_cache=False)

    def _load(self, use_earnings_cache: bool):
        ticker = self._ticker
        if ticker is None:
            return

        alpaca = self._fetcher.alpaca

        def job():
            earnings = (
                get_earnings_dates(ticker)
                if use_earnings_cache else refresh_earnings_dates(ticker)
            )
            bars = _ensure_bars_cover(ticker, earnings, alpaca)
            if bars is None or bars.empty:
                bars = pd.DataFrame(columns=["timestamp", "close"])

            # Chain loader that ONLY hits cache — avoids any network during
            # the batch. Pre-Feb-2024 dates simply return None.
            def chain_loader(tkr: str, day: date):
                return cache.read_chain(tkr, day)

            table = build_earnings_table(
                ticker, bars, earnings, chain_loader=chain_loader,
            )
            return earnings, table

        worker = Worker(job)
        worker.signals.finished.connect(self._on_loaded)
        worker.signals.failed.connect(self._on_failed)
        self._pool.start(worker)

    def _on_failed(self, msg: str):
        self.status.setText(f"Earnings load failed: {msg.splitlines()[0]}")

    def _on_loaded(self, payload):
        earnings, table = payload
        if table is None or table.empty:
            self.status.setText(
                f"No earnings history available for {self._ticker}."
            )
            self.table.setRowCount(0)
            self._clear_histogram()
            return

        n = len(table)
        n_impl = int(table["implied_straddle"].notna().sum())
        n_real = int(table["realized"].notna().sum())
        self.status.setText(
            f"{self._ticker}: {n} earnings events — "
            f"{n_real} with realized moves, {n_impl} with options data. "
            f"Backfill older chains to extend implied-move coverage."
        )
        self._populate_table(table)
        self._draw_histogram(table["realized"].dropna().to_numpy())

    # ---- rendering --------------------------------------------------------

    def _populate_table(self, df: pd.DataFrame):
        self.table.setRowCount(len(df))
        for r, row in enumerate(df.itertuples(index=False)):
            items = [
                QTableWidgetItem(str(row.date)),
                QTableWidgetItem(row.time_of_day or ""),
                QTableWidgetItem(_fmt_pct(row.implied_straddle)),
                QTableWidgetItem(_fmt_pct(row.implied_jump)),
                QTableWidgetItem(_fmt_pct(row.realized)),
                QTableWidgetItem(_fmt_signed_pct(row.spread)),
            ]
            # Color the spread column to highlight vol-crush winners/losers.
            # spread = realized - implied; negative means implied > realized
            # (i.e. selling vol into earnings would have paid off).
            if np.isfinite(row.spread):
                color = QColor("#6aff9a") if row.spread < 0 else QColor("#ff6a6a")
                items[-1].setForeground(color)
            for c, it in enumerate(items):
                if c > 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, it)

    def _clear_histogram(self):
        if self._hist_bar is not None:
            self.hist_plot.removeItem(self._hist_bar)
            self._hist_bar = None
        for ln in self._hist_lines:
            self.hist_plot.removeItem(ln)
        self._hist_lines = []

    def _draw_histogram(self, values: np.ndarray):
        self._clear_histogram()
        if values.size == 0:
            return
        pct = values * 100.0
        # Freedman-Diaconis would be nicer but given ~8-20 observations a
        # fixed bin count is visually more readable.
        n_bins = max(5, min(15, int(np.ceil(np.sqrt(values.size))) + 2))
        counts, edges = np.histogram(pct, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = float(edges[1] - edges[0]) * 0.9
        self._hist_bar = pg.BarGraphItem(
            x=centers, height=counts, width=width,
            brush=pg.mkBrush("#4aa3ff"), pen=pg.mkPen("#ffffff", width=1),
        )
        self.hist_plot.addItem(self._hist_bar)

        mean_v = float(np.mean(pct))
        med_v = float(np.median(pct))
        for pos, color, label in (
            (mean_v, "#ffff00", f"mean {mean_v:.1f}%"),
            (med_v, "#ff8c00", f"median {med_v:.1f}%"),
        ):
            ln = pg.InfiniteLine(
                pos=pos, angle=90,
                pen=pg.mkPen(color, width=2, style=Qt.DashLine),
                label=label,
                labelOpts={"position": 0.95, "color": color, "fill": "#111"},
            )
            self.hist_plot.addItem(ln)
            self._hist_lines.append(ln)
