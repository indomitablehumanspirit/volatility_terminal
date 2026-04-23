"""VRP tab: ATM-IV timeseries at fixed DTE vs rolling realized vol."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ...analytics.iv_timeseries import build_iv_timeseries
from ...analytics.realized import close_to_close
from ...analytics.vrp import compute_vrp
from ...data import cache


class VrpTab(QWidget):
    rebuild_requested = pyqtSignal(str, int)  # ticker, dte

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("DTE:"))
        self.dte_spin = QSpinBox()
        self.dte_spin.setRange(7, 365)
        self.dte_spin.setValue(30)
        ctrl.addWidget(self.dte_spin)
        self.rebuild_btn = QPushButton("Rebuild series")
        self.rebuild_btn.clicked.connect(self._on_rebuild)
        ctrl.addWidget(self.rebuild_btn)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#111")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()
        self.plot.setLabel("left", "Vol (%)")
        self.plot.setLabel("bottom", "Date")
        self.plot.setAxisItems({"bottom": pg.DateAxisItem()})
        root.addWidget(self.plot, 3)

        self.vrp_plot = pg.PlotWidget()
        self.vrp_plot.setBackground("#111")
        self.vrp_plot.showGrid(x=True, y=True, alpha=0.3)
        self.vrp_plot.setLabel("left", "VRP (IV-RV) %")
        self.vrp_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        self.vrp_plot.setXLink(self.plot)
        root.addWidget(self.vrp_plot, 2)

        self.iv_curve = self.plot.plot(
            [], [], pen=pg.mkPen("#4aa3ff", width=2), name="ATM IV")
        self.rv_curve = self.plot.plot(
            [], [], pen=pg.mkPen("#ff6a6a", width=2), name="Realized Vol 30d")
        self.vrp_bar = pg.BarGraphItem(x=[], height=[], width=60 * 60 * 12,
                                       brush=pg.mkBrush("#6aff9a"))
        self.vrp_plot.addItem(self.vrp_bar)
        self.vrp_plot.addItem(pg.InfiniteLine(pos=0, angle=0,
                                              pen=pg.mkPen("#888", style=Qt.DashLine)))

        self._ticker: str | None = None

    def set_ticker(self, ticker: str):
        self._ticker = ticker.upper()
        self.refresh()

    def _on_rebuild(self):
        if self._ticker:
            self.rebuild_requested.emit(self._ticker, self.dte_spin.value())

    def refresh(self):
        if not self._ticker:
            return
        dte = self.dte_spin.value()
        try:
            iv_ts = build_iv_timeseries(self._ticker, dte)
        except Exception:
            iv_ts = pd.DataFrame(columns=["date", "atm_iv", "spot"])
        under = cache.read_underlying(self._ticker)
        if under is None or under.empty:
            rv = pd.Series(dtype=float)
        else:
            under = under.copy()
            under["date"] = pd.to_datetime(under["timestamp"]).dt.tz_localize(None).dt.normalize()
            under = under.drop_duplicates("date").set_index("date")
            rv = close_to_close(under["close"].astype(float), window=dte)

        merged = compute_vrp(iv_ts, rv)
        if merged.empty:
            self.iv_curve.setData([], [])
            self.rv_curve.setData([], [])
            self.vrp_bar.setOpts(x=[], height=[])
            return
        xs = pd.to_datetime(merged["date"]).view("int64") // 1_000_000_000
        self.iv_curve.setData(xs.to_numpy(), (merged["atm_iv"] * 100).to_numpy())
        rv_mask = merged["rv"].notna()
        self.rv_curve.setData(xs[rv_mask].to_numpy(),
                              (merged.loc[rv_mask, "rv"] * 100).to_numpy())
        vrp_mask = merged["vrp"].notna()
        self.vrp_bar.setOpts(
            x=xs[vrp_mask].to_numpy(),
            height=(merged.loc[vrp_mask, "vrp"] * 100).to_numpy(),
            width=60 * 60 * 12,
        )
