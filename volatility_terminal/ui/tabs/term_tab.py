"""Term structure tab: ATM IV vs days-to-expiry, with multi-dataset comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...analytics.forward_vol import forward_vol
from ...analytics.term import term_structure
from ..comparison_panel import ComparisonPanel

# Colors for comparison datasets (primary uses white/blue/red)
_COMP_COLORS = [
    "#ff8c00", "#00e676", "#ce93d8", "#ffff00",
    "#00bcd4", "#ff5722", "#8bc34a", "#e91e63",
]

_FWD_COLS = ["T1→T2", "DTE₁→DTE₂", "Fwd Vol %"]


class _DTEDateAxis(pg.AxisItem):
    """Log-scale axis over days-to-expiry whose ticks render as calendar dates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ref: pd.Timestamp | None = None

    def set_reference_date(self, ref):
        self._ref = pd.Timestamp(ref) if ref is not None else None
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self._ref is None:
            # fall back to "N d"
            return [f"{10 ** v:.0f}d" for v in values]
        out = []
        for v in values:
            days = 10 ** v
            d = (self._ref + pd.Timedelta(days=days)).date()
            out.append(d.strftime("%Y-%m-%d"))
        return out


def _fmt(x, digits=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


class TermTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)

        self._date_axis = _DTEDateAxis(orientation="bottom")
        self.plot = pg.PlotWidget(axisItems={"bottom": self._date_axis})
        self.plot.setBackground("#111")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Implied Vol (%)")
        self.plot.setLabel("bottom", "Expiry")
        self.plot.setLogMode(x=True, y=False)
        self.plot.addLegend()
        content.addWidget(self.plot, 1)

        right = QVBoxLayout()
        right.setContentsMargins(2, 0, 2, 0)
        right.setSpacing(4)

        self.fwd_table = QTableWidget(0, len(_FWD_COLS))
        self.fwd_table.setHorizontalHeaderLabels(_FWD_COLS)
        self.fwd_table.verticalHeader().setVisible(False)
        self.fwd_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fwd_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.fwd_table.setAlternatingRowColors(True)
        self.fwd_table.setShowGrid(False)
        self.fwd_table.setStyleSheet(
            "QTableWidget { background-color: #111; color: #ddd;"
            " alternate-background-color: #1a1a1a; gridline-color: #333; }"
            "QHeaderView::section { background-color: #222; color: #ccc;"
            " border: 0px; padding: 3px; }"
        )
        header = self.fwd_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(False)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.fwd_table.setFont(mono)
        self.fwd_table.setFixedWidth(210)
        right.addWidget(self.fwd_table, 1)

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

        primary_ts: pd.DataFrame | None = None

        if self._primary is not None:
            ticker, day, chain = self._primary
            self._date_axis.set_reference_date(day)
            if chain is not None and not chain.empty:
                ts = term_structure(chain)
                primary_ts = ts
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

        self._update_fwd_table(primary_ts)

    def _update_fwd_table(self, ts: pd.DataFrame | None):
        self.fwd_table.setRowCount(0)
        if ts is None or ts.empty:
            return
        fv = forward_vol(ts)
        if fv.empty:
            return
        self.fwd_table.setRowCount(len(fv))
        arb_color = QColor("#ff6a6a")
        for r, row in fv.reset_index(drop=True).iterrows():
            e1 = pd.Timestamp(row["expiry_1"]).date()
            e2 = pd.Timestamp(row["expiry_2"]).date()
            vals = [
                f"{e1.strftime('%m-%d')}→{e2.strftime('%m-%d')}",
                f"{row['dte_1']:.0f}→{row['dte_2']:.0f}",
                _fmt(row["fwd_vol"] * 100, 2),
            ]
            arb = not np.isfinite(row["fwd_vol"])
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if arb:
                    item.setForeground(arb_color)
                self.fwd_table.setItem(r, col, item)
