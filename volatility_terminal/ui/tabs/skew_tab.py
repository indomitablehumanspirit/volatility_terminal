"""Skew tab: IV vs log-moneyness, multi-expiry, multi-dataset comparison."""
from __future__ import annotations

import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from ...analytics.skew import skew_for_expiry
from ..comparison_panel import ComparisonPanel

# Colormaps per comparison dataset (primary uses viridis)
_COMP_CMAPS = ["plasma", "inferno", "magma", "cividis"]


def _nearest_expiry(chain: pd.DataFrame, target_dte: float):
    """Return the expiry in chain whose DTE is closest to target_dte days."""
    per_exp = chain.groupby("expiry")["tau"].first()
    if per_exp.empty:
        return None
    return (per_exp * 365.25 - target_dte).abs().idxmin()


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
        self.plot.addItem(
            pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#888", style=Qt.DashLine))
        )
        root.addWidget(self.plot, 1)

        right = QVBoxLayout()
        right.setContentsMargins(2, 0, 2, 0)
        right.setSpacing(4)

        self.expiry_list = QListWidget()
        self.expiry_list.setSelectionMode(QListWidget.MultiSelection)
        self.expiry_list.setFixedWidth(200)
        self.expiry_list.itemSelectionChanged.connect(self._redraw)
        right.addWidget(self.expiry_list, 1)

        self.comparison_panel = ComparisonPanel()
        self.comparison_panel.setFixedWidth(200)
        right.addWidget(self.comparison_panel)

        root.addLayout(right)

        self._primary: tuple | None = None          # (ticker, day, chain)
        self._comparisons: dict[int, tuple] = {}    # entry_id -> (ticker, day, chain)
        self._curves: list = []

    def set_chain(self, ticker: str, day, chain: pd.DataFrame):
        self._primary = (str(ticker).upper(), day, chain)
        self.expiry_list.blockSignals(True)
        self.expiry_list.clear()
        if chain is None or chain.empty:
            self.expiry_list.blockSignals(False)
            self._redraw()
            return
        per_exp = chain.groupby("expiry")["tau"].first().sort_values()
        preselect = set(per_exp.head(5).index)
        for exp, tau in per_exp.items():
            item = QListWidgetItem(
                f"{pd.Timestamp(exp).date()}  ({tau * 365.25:.0f}d)"
            )
            item.setData(Qt.UserRole, exp)
            self.expiry_list.addItem(item)
            if exp in preselect:
                item.setSelected(True)
        self.expiry_list.blockSignals(False)
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

        if self._primary is None:
            return
        ticker, day, chain = self._primary
        if chain is None or chain.empty:
            return

        selected = [
            self.expiry_list.item(i).data(Qt.UserRole)
            for i in range(self.expiry_list.count())
            if self.expiry_list.item(i).isSelected()
        ]
        if not selected:
            return

        cmap_primary = pg.colormap.get("viridis")
        n = max(len(selected), 1)
        sorted_eids = sorted(self._comparisons)
        comp_cmaps = {
            eid: pg.colormap.get(_COMP_CMAPS[i % len(_COMP_CMAPS)])
            for i, eid in enumerate(sorted_eids)
        }

        per_exp_primary = chain.groupby("expiry")["tau"].first()

        for i, exp in enumerate(selected):
            tau_val = per_exp_primary.get(exp)
            if tau_val is None:
                continue
            target_dte = tau_val * 365.25
            color_pos = 0.15 + 0.70 * (i / max(n - 1, 1))

            # Primary curve
            df = skew_for_expiry(chain, exp)
            if not df.empty:
                color = cmap_primary.map(color_pos, mode="qcolor")
                c = self.plot.plot(
                    df["log_moneyness"].to_numpy(),
                    (df["iv"] * 100).to_numpy(),
                    pen=pg.mkPen(color, width=2),
                    symbol="o", symbolSize=5, symbolBrush=color,
                    name=f"{ticker} {day} | {target_dte:.0f}d",
                )
                self._curves.append(c)

            # Comparison curves — matched to nearest DTE in each dataset
            for eid in sorted_eids:
                comp_ticker, comp_day, comp_chain = self._comparisons[eid]
                if comp_chain is None or comp_chain.empty:
                    continue
                matched_exp = _nearest_expiry(comp_chain, target_dte)
                if matched_exp is None:
                    continue
                df2 = skew_for_expiry(comp_chain, matched_exp)
                if df2.empty:
                    continue
                matched_tau = comp_chain.groupby("expiry")["tau"].first().get(matched_exp)
                matched_dte = matched_tau * 365.25 if matched_tau is not None else target_dte
                color = comp_cmaps[eid].map(color_pos, mode="qcolor")
                c2 = self.plot.plot(
                    df2["log_moneyness"].to_numpy(),
                    (df2["iv"] * 100).to_numpy(),
                    pen=pg.mkPen(color, width=2, style=2),
                    symbol="s", symbolSize=5, symbolBrush=color,
                    name=f"{comp_ticker} {comp_day} | {matched_dte:.0f}d",
                )
                self._curves.append(c2)
