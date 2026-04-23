"""Term structure tab: ATM IV vs days-to-expiry, with multi-dataset comparison."""
from __future__ import annotations

import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...analytics.term import term_structure
from ..comparison_panel import ComparisonPanel

# Colors for comparison datasets (primary uses white/blue/red)
_COMP_COLORS = [
    "#ff8c00", "#00e676", "#ce93d8", "#ffff00",
    "#00bcd4", "#ff5722", "#8bc34a", "#e91e63",
]


class TermTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#111")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Implied Vol (%)")
        self.plot.setLabel("bottom", "Days to expiry")
        self.plot.setLogMode(x=True, y=False)
        self.plot.addLegend()
        content.addWidget(self.plot, 1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addStretch(1)
        self.comparison_panel = ComparisonPanel()
        self.comparison_panel.setFixedWidth(210)
        right.addWidget(self.comparison_panel)
        content.addLayout(right)

        root.addLayout(content, 1)

        self._primary: tuple | None = None          # (ticker, day, chain)
        self._comparisons: dict[int, tuple] = {}    # entry_id -> (ticker, day, chain)
        self._curves: list = []

    def set_chain(self, ticker: str, day, chain: pd.DataFrame):
        self._primary = (str(ticker).upper(), day, chain)
        self._redraw()

    def add_comparison(self, entry_id: int, ticker: str, day, chain: pd.DataFrame):
        self._comparisons[entry_id] = (str(ticker).upper(), day, chain)
        self._redraw()

    def remove_comparison(self, entry_id: int):
        self._comparisons.pop(entry_id, None)
        self._redraw()

    def _redraw(self):
        for c in self._curves:
            self.plot.removeItem(c)
        self._curves = []
        legend = self.plot.getPlotItem().legend
        if legend is not None:
            legend.clear()

        if self._primary is not None:
            ticker, day, chain = self._primary
            if chain is not None and not chain.empty:
                ts = term_structure(chain)
                if not ts.empty:
                    days = (ts["tau"] * 365.25).to_numpy()
                    prefix = f"{ticker} {day}"
                    c1 = self.plot.plot(
                        days, (ts["atm_iv"] * 100).to_numpy(),
                        pen=pg.mkPen("#ffffff", width=2),
                        symbol="o", symbolSize=7, symbolBrush="#ffffff",
                        name=f"{prefix} ATM",
                    )
                    c2 = self.plot.plot(
                        days, (ts["call_iv"] * 100).to_numpy(),
                        pen=pg.mkPen("#4aa3ff", width=1, style=2),
                        symbol="t1", symbolSize=6, symbolBrush="#4aa3ff",
                        name=f"{prefix} Call",
                    )
                    c3 = self.plot.plot(
                        days, (ts["put_iv"] * 100).to_numpy(),
                        pen=pg.mkPen("#ff6a6a", width=1, style=2),
                        symbol="t", symbolSize=6, symbolBrush="#ff6a6a",
                        name=f"{prefix} Put",
                    )
                    self._curves.extend([c1, c2, c3])

        # Comparison datasets: ATM only to avoid visual clutter
        for entry_id, (comp_ticker, comp_day, comp_chain) in sorted(
            self._comparisons.items()
        ):
            if comp_chain is None or comp_chain.empty:
                continue
            ts = term_structure(comp_chain)
            if ts.empty:
                continue
            days = (ts["tau"] * 365.25).to_numpy()
            color = _COMP_COLORS[entry_id % len(_COMP_COLORS)]
            c = self.plot.plot(
                days, (ts["atm_iv"] * 100).to_numpy(),
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=7, symbolBrush=color,
                name=f"{comp_ticker} {comp_day}",
            )
            self._curves.append(c)
