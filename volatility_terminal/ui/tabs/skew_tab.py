"""Skew tab: IV vs log-moneyness, multi-expiry, multi-dataset comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...analytics.skew import skew_for_expiry
from ...analytics.skew_metrics import delta_skew_metrics
from ..comparison_panel import ComparisonPanel

# Colormaps per comparison dataset (primary uses viridis)
_COMP_CMAPS = ["plasma", "inferno", "magma", "cividis"]

_METRIC_COLS = ["Dataset", "Expiry", "DTE", "ATM IV %",
                "25Δ Put IV %", "25Δ Call IV %", "RR (vol pts)", "BF (vol pts)"]


def _nearest_expiry(chain: pd.DataFrame, target_dte: float):
    """Return the expiry in chain whose DTE is closest to target_dte days."""
    per_exp = chain.groupby("expiry")["tau"].first()
    if per_exp.empty:
        return None
    return (per_exp * 365.25 - target_dte).abs().idxmin()


def _fmt(x, digits=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


class SkewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#111")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Implied Vol (%)")
        self.plot.setLabel("bottom", "log(Strike / Forward)")
        self.plot.addLegend()
        self.plot.addItem(
            pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#888", style=Qt.DashLine))
        )
        left.addWidget(self.plot, 1)

        self.metrics_table = QTableWidget(0, len(_METRIC_COLS))
        self.metrics_table.setHorizontalHeaderLabels(_METRIC_COLS)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setShowGrid(False)
        self.metrics_table.setStyleSheet(
            "QTableWidget { background-color: #111; color: #ddd;"
            " alternate-background-color: #1a1a1a; gridline-color: #333; }"
            "QHeaderView::section { background-color: #222; color: #ccc;"
            " border: 0px; padding: 3px; }"
        )
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(False)
        self.metrics_table.setFixedHeight(180)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.metrics_table.setFont(mono)
        left.addWidget(self.metrics_table)

        root.addLayout(left, 1)

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
        self._metrics_cache: dict[int, pd.DataFrame] = {}

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

    def _metrics_for(self, chain: pd.DataFrame) -> pd.DataFrame:
        if chain is None or chain.empty:
            return pd.DataFrame()
        key = id(chain)
        df = self._metrics_cache.get(key)
        if df is None:
            df = delta_skew_metrics(chain)
            self._metrics_cache[key] = df
        return df

    def _redraw(self):
        for c in self._curves:
            self.plot.removeItem(c)
        self._curves = []
        legend = self.plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        self.metrics_table.setRowCount(0)

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
        pm = self._metrics_for(chain)
        primary_metrics = pm.set_index("expiry") if not pm.empty else pd.DataFrame()

        table_rows: list[tuple] = []

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

            if not primary_metrics.empty and exp in primary_metrics.index:
                row = primary_metrics.loc[exp]
                label_color = cmap_primary.map(color_pos, mode="qcolor")
                table_rows.append((
                    f"{ticker} {day}", pd.Timestamp(exp).date(), row, label_color,
                ))

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

                comp_metrics = self._metrics_for(comp_chain)
                if not comp_metrics.empty:
                    comp_metrics_i = comp_metrics.set_index("expiry")
                    if matched_exp in comp_metrics_i.index:
                        row = comp_metrics_i.loc[matched_exp]
                        table_rows.append((
                            f"{comp_ticker} {comp_day}",
                            pd.Timestamp(matched_exp).date(), row, color,
                        ))

        self.metrics_table.setRowCount(len(table_rows))
        for r, (label, exp_date, m, color) in enumerate(table_rows):
            atm = float(m["atm_iv"]) if np.isfinite(m["atm_iv"]) else float("nan")
            put25 = float(m["put25_iv"]) if np.isfinite(m["put25_iv"]) else float("nan")
            call25 = float(m["call25_iv"]) if np.isfinite(m["call25_iv"]) else float("nan")
            rr = float(m["rr_25d"]) if np.isfinite(m["rr_25d"]) else float("nan")
            bf = float(m["bf_25d"]) if np.isfinite(m["bf_25d"]) else float("nan")
            vals = [
                label,
                str(exp_date),
                f"{m['dte']:.0f}",
                _fmt(atm * 100, 2),
                _fmt(put25 * 100, 2),
                _fmt(call25 * 100, 2),
                _fmt(rr * 100, 2),
                _fmt(bf * 100, 2),
            ]
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 0 and isinstance(color, QColor):
                    item.setForeground(color)
                self.metrics_table.setItem(r, col, item)
