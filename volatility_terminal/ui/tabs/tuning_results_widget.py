"""Tuning results display: header, best params, IS/OOS equity, sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...analytics.tuning import TuningResult


_OBJ_LABEL = {
    "profit_factor": "Profit factor",
    "sharpe": "Sharpe",
    "total_pnl": "Total PnL ($)",
}


def _fmt_score(v: float, obj: str) -> str:
    if not np.isfinite(v):
        return "n/a"
    if obj == "total_pnl":
        return f"${v:,.0f}"
    return f"{v:.3f}"


class TuningResultsWidget(QWidget):
    params_apply_requested = pyqtSignal(dict)

    def __init__(self, result: TuningResult, parent=None):
        super().__init__(parent)
        self._result = result
        self._best_params = self._extract_best_params(result)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        obj = result.objective
        obj_lbl = _OBJ_LABEL.get(obj, obj)

        # --- Header summary ---
        is_s = _fmt_score(result.best_is_score, obj)
        oos_s = _fmt_score(result.oos_score, obj)
        if (np.isfinite(result.best_is_score) and result.best_is_score != 0
                and np.isfinite(result.oos_score)):
            ratio = result.oos_score / result.best_is_score
        else:
            ratio = float("nan")

        is_a, is_b = result.is_dates
        oos_a, oos_b = result.oos_dates
        ratio_pct = "n/a" if not np.isfinite(ratio) else f"{ratio * 100:.0f}%"
        if np.isfinite(ratio):
            if ratio < 0.6:
                col = "#ef6262"
            elif ratio < 0.8:
                col = "#ffd166"
            else:
                col = "#5fcf5f"
        else:
            col = "#bbb"

        header = QLabel(
            f"IS [{is_a} → {is_b}] {obj_lbl}: {is_s}  |  "
            f"OOS [{oos_a} → {oos_b}] {obj_lbl}: {oos_s}  |  "
            f"<span style='color:{col};'>IS/OOS ratio: {ratio_pct}</span>  |  "
            f"Trials: {len(result.all_trials)}"
        )
        header.setTextFormat(Qt.RichText)
        header.setStyleSheet("color: #ddd; padding: 4px;")
        header.setWordWrap(True)
        root.addWidget(header)

        # --- Warning ---
        if np.isfinite(ratio) and ratio < 0.6:
            warn = QLabel(
                "⚠ Large IS/OOS gap — parameters may be overfit to the IS period.")
            warn.setStyleSheet("color: #ef6262; padding: 2px 4px;")
            root.addWidget(warn)

        # --- Best params table + apply button ---
        params_box = QGroupBox("Best parameters")
        pb = QVBoxLayout(params_box)
        self.params_table = QTableWidget(len(self._best_params), 2)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Best value"])
        self.params_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.params_table.verticalHeader().setVisible(False)
        self.params_table.setEditTriggers(QTableWidget.NoEditTriggers)
        labels = self._param_labels(result)
        for r, (k, v) in enumerate(self._best_params.items()):
            self.params_table.setItem(r, 0, QTableWidgetItem(labels.get(k, k)))
            if isinstance(v, float):
                txt = f"{v:.4g}"
            else:
                txt = str(v)
            self.params_table.setItem(r, 1, QTableWidgetItem(txt))
        self.params_table.setMaximumHeight(180)
        pb.addWidget(self.params_table)
        apply_row = QHBoxLayout()
        apply_row.addStretch(1)
        self.apply_btn = QPushButton("Apply to backtest")
        self.apply_btn.clicked.connect(
            lambda: self.params_apply_requested.emit(dict(self._best_params)))
        apply_row.addWidget(self.apply_btn)
        pb.addLayout(apply_row)
        root.addWidget(params_box)

        # --- IS / OOS equity curves ---
        eq_row = QHBoxLayout()
        self.is_plot = self._make_eq_plot(
            result.is_result, "IS equity (in-sample — optimizer saw this)",
            "#4aa3ff")
        self.oos_plot = self._make_eq_plot(
            result.oos_result,
            "OOS equity (holdout — optimizer never saw this)", "#5fcf5f")
        eq_row.addWidget(self.is_plot)
        eq_row.addWidget(self.oos_plot)
        root.addLayout(eq_row)

        # --- Sensitivity charts ---
        sens_box = QGroupBox("Parameter sensitivity (IS sweep, OOS marker)")
        sb = QVBoxLayout(sens_box)
        self._sens_layout = QVBoxLayout()
        self._build_sensitivity_charts(result, labels, obj_lbl)
        sb.addLayout(self._sens_layout)
        root.addWidget(sens_box)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_best_params(result: TuningResult) -> dict:
        if not result.all_trials:
            return {}
        best = max(result.all_trials, key=lambda t: t[1])
        return dict(best[0])

    @staticmethod
    def _param_labels(result: TuningResult) -> dict[str, str]:
        # The widget receives only the final best dict; labels come from sensitivity
        # keys. Fall back to the raw name. Caller-provided labels would be ideal,
        # but TuningResult doesn't carry them — store a minimal mapping here.
        return {}

    # ------------------------------------------------------------------

    def _make_eq_plot(self, res, title: str, color: str) -> QWidget:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        plot = pg.PlotWidget()
        plot.setBackground("#111")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("left", "Cum PnL ($)")
        plot.setAxisItems({"bottom": pg.DateAxisItem()})
        plot.setMinimumHeight(150)
        plot.setMaximumHeight(180)
        plot.addItem(pg.InfiniteLine(pos=0, angle=0,
                     pen=pg.mkPen("#555", style=Qt.DashLine)))
        if res is not None and not res.error and not res.equity_curve.empty:
            eq = res.equity_curve
            xs = pd.to_datetime(eq["date"]).view("int64") // 1_000_000_000
            plot.plot(xs.to_numpy(), eq["cum_pnl"].to_numpy(),
                      pen=pg.mkPen(color, width=2))
        lay.addWidget(plot)
        return box

    # ------------------------------------------------------------------

    def _build_sensitivity_charts(self, result: TuningResult,
                                   labels: dict[str, str], obj_lbl: str):
        eps = 1e-12
        for name, (xs, ys) in result.sensitivity.items():
            box = QGroupBox(labels.get(name, name))
            inner = QVBoxLayout(box)
            inner.setContentsMargins(4, 4, 4, 4)
            plot = pg.PlotWidget()
            plot.setBackground("#111")
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setLabel("left", obj_lbl)
            plot.setLabel("bottom", labels.get(name, name))
            plot.setMinimumHeight(140)
            plot.setMaximumHeight(180)

            # Filter inf scores for plotting
            ys_plot = np.where(np.isfinite(ys), ys, np.nan)
            valid = np.isfinite(ys_plot)
            if valid.any():
                plot.plot(xs[valid], ys_plot[valid],
                          pen=pg.mkPen("#4aa3ff", width=2),
                          symbol="o", symbolSize=5,
                          symbolBrush=pg.mkBrush("#4aa3ff"))

            # Vertical dashed line at optimal
            best_x = self._best_params.get(name)
            if best_x is not None:
                vline = pg.InfiniteLine(
                    pos=float(best_x), angle=90,
                    pen=pg.mkPen("#ffd166", width=1, style=Qt.DashLine))
                plot.addItem(vline)

            # OOS marker
            oos_y = result.oos_sensitivity_point.get(name)
            if best_x is not None and oos_y is not None and np.isfinite(oos_y):
                sc = pg.ScatterPlotItem(
                    [float(best_x)], [float(oos_y)],
                    symbol="o", size=12,
                    pen=pg.mkPen("#ef6262", width=2),
                    brush=pg.mkBrush("#ef6262"))
                plot.addItem(sc)
                txt = pg.TextItem("OOS", color="#ef6262", anchor=(0, 1))
                txt.setPos(float(best_x), float(oos_y))
                plot.addItem(txt)

            inner.addWidget(plot)

            # Interpretation label
            finite = ys_plot[valid] if valid.any() else np.array([])
            if finite.size >= 2:
                rng = float(np.nanmax(finite) - np.nanmin(finite))
                step = float(np.nanmax(np.abs(np.diff(finite))))
                slope = step / (rng + eps)
            else:
                slope = 0.0
            if slope > 0.4:
                txt = "Steep curve — high sensitivity to this parameter. " \
                      "Strategy fragile to non-stationarity."
                col = "#ef6262"
            elif slope < 0.15:
                txt = "Flat curve — low sensitivity. Robust to regime drift."
                col = "#5fcf5f"
            else:
                txt = "Moderate sensitivity."
                col = "#ffd166"
            interp = QLabel(txt)
            interp.setStyleSheet(f"color: {col}; padding: 2px 4px;")
            interp.setWordWrap(True)
            inner.addWidget(interp)

            self._sens_layout.addWidget(box)

    # ------------------------------------------------------------------

    def set_param_labels(self, labels: dict[str, str]):
        """Inject human-readable labels for the parameter table + sensitivity
        charts. Called by the host tab after construction."""
        for r in range(self.params_table.rowCount()):
            key_item = self.params_table.item(r, 0)
            if key_item is None:
                continue
            key = key_item.text()
            # Re-key from internal name → label only if the row is still showing
            # the raw name (initial render). Look up by reverse-scanning the
            # best_params dict by index.
        # Rebuild table cells using internal names → labels
        keys = list(self._best_params.keys())
        for r, k in enumerate(keys):
            self.params_table.item(r, 0).setText(labels.get(k, k))
        # Rebuild sensitivity charts with labels — safer to rebuild fresh.
        # Clear existing
        while self._sens_layout.count():
            item = self._sens_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        obj_lbl = _OBJ_LABEL.get(self._result.objective, self._result.objective)
        self._build_sensitivity_charts(self._result, labels, obj_lbl)
