"""Skew tab: one IV-vs-log-moneyness curve per expiry, toggleable."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from ...analytics.skew import skew_for_expiry


class SkewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#111")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Implied Vol (%)")
        self.plot.setLabel("bottom", "log(Strike / Forward)")
        self.plot.addLegend()
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90,
                                          pen=pg.mkPen("#888", style=Qt.DashLine)))
        root.addWidget(self.plot, 1)

        self.expiry_list = QListWidget()
        self.expiry_list.setSelectionMode(QListWidget.MultiSelection)
        self.expiry_list.setFixedWidth(180)
        self.expiry_list.itemSelectionChanged.connect(self._redraw)
        root.addWidget(self.expiry_list)

        self._chain: pd.DataFrame | None = None
        self._curves: list = []

    def set_chain(self, chain: pd.DataFrame):
        self._chain = chain
        self.expiry_list.blockSignals(True)
        self.expiry_list.clear()
        if chain is None or chain.empty:
            self.expiry_list.blockSignals(False)
            self._redraw()
            return
        per_exp = chain.groupby("expiry")["tau"].first().sort_values()
        # preselect the first 5 shortest-dated
        preselect = set(per_exp.head(5).index)
        for exp, tau in per_exp.items():
            item = QListWidgetItem(f"{pd.Timestamp(exp).date()}  ({tau*365.25:.0f}d)")
            item.setData(Qt.UserRole, exp)
            self.expiry_list.addItem(item)
            if exp in preselect:
                item.setSelected(True)
        self.expiry_list.blockSignals(False)
        self._redraw()

    def _redraw(self):
        for c in self._curves:
            self.plot.removeItem(c)
        self._curves = []
        if self._chain is None or self._chain.empty:
            return
        selected = [self.expiry_list.item(i).data(Qt.UserRole)
                    for i in range(self.expiry_list.count())
                    if self.expiry_list.item(i).isSelected()]
        cmap = pg.colormap.get("viridis")
        n = max(len(selected), 1)
        for i, exp in enumerate(selected):
            df = skew_for_expiry(self._chain, exp)
            if df.empty:
                continue
            color = cmap.map(0.15 + 0.70 * (i / max(n - 1, 1)), mode="qcolor")
            curve = self.plot.plot(
                df["log_moneyness"].to_numpy(),
                (df["iv"] * 100).to_numpy(),
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=5, symbolBrush=color,
                name=f"{pd.Timestamp(exp).date()}",
            )
            self._curves.append(curve)
