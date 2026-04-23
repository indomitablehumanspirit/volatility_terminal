"""Term structure tab: ATM IV vs days-to-expiry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ...analytics.term import term_structure


class TermTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#111")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Implied Vol (%)")
        self.plot.setLabel("bottom", "Days to expiry")
        self.plot.setLogMode(x=True, y=False)
        self.plot.addLegend()
        layout.addWidget(self.plot)

        self.atm_curve = self.plot.plot(
            [], [], pen=pg.mkPen("#ffffff", width=2),
            symbol="o", symbolSize=7, symbolBrush="#ffffff", name="ATM IV"
        )
        self.call_curve = self.plot.plot(
            [], [], pen=pg.mkPen("#4aa3ff", width=1, style=2),
            symbol="t1", symbolSize=6, symbolBrush="#4aa3ff", name="Call ATM"
        )
        self.put_curve = self.plot.plot(
            [], [], pen=pg.mkPen("#ff6a6a", width=1, style=2),
            symbol="t", symbolSize=6, symbolBrush="#ff6a6a", name="Put ATM"
        )

    def set_chain(self, chain: pd.DataFrame):
        ts = term_structure(chain)
        if ts.empty:
            for c in (self.atm_curve, self.call_curve, self.put_curve):
                c.setData([], [])
            return
        days = (ts["tau"] * 365.25).to_numpy()
        self.atm_curve.setData(days, (ts["atm_iv"] * 100).to_numpy())
        self.call_curve.setData(days, (ts["call_iv"] * 100).to_numpy())
        self.put_curve.setData(days, (ts["put_iv"] * 100).to_numpy())
