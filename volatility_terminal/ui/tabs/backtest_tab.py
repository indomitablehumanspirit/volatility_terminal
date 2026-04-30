"""Backtest tab — composable signal-driven backtests over historical chains.

Layout (top to bottom):
  - Toolbar: ticker, date range, structure, direction, qty, run, save/load.
  - Leg-spec rows (dynamic; rebuilt when structure kind changes).
  - Structural exits + hedge config (one row each).
  - Splitter:
      Left:  Entry rule + Exit rule.
      Right: Signal preview chart with threshold + entry/exit/hedge markers.
  - Equity curve + summary + trades table.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import QDate, QSettings, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...analytics.backtest_config import BacktestConfig
from ...analytics.signals import (
    Condition, RuleConfig, signal_from_dict,
)
from ...analytics.simulation import HedgeConfig
from ...analytics.structures import (
    StructureKind, StructureParams, default_legs,
)
from ..rule_widget import RuleWidget
from ..signal_library import SignalLibrary


SETTINGS_ORG = "VolatilityTerminal"
SETTINGS_APP = "VolatilityTerminal"
CONFIGS_KEY = "backtest/saved_configs"


STRUCTURES: list[tuple[str, StructureKind]] = [
    ("Naked call", "naked_call"),
    ("Naked put", "naked_put"),
    ("Straddle", "straddle"),
    ("Strangle", "strangle"),
    ("Vertical (call)", "vertical_call"),
    ("Vertical (put)", "vertical_put"),
    ("Calendar (call)", "calendar_call"),
    ("Calendar (put)", "calendar_put"),
    ("Butterfly (call)", "butterfly_call"),
    ("Butterfly (put)", "butterfly_put"),
    ("Iron condor", "iron_condor"),
]


class _LegSpecRow(QWidget):
    changed = pyqtSignal()

    def __init__(self, idx: int, right: str, side: str,
                 dte: int, delta_target: float | None, parent=None):
        super().__init__(parent)
        self.right = right
        self.side = side
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        title = QLabel(f"Leg {idx+1}: {side} {right}")
        title.setMinimumWidth(120)
        layout.addWidget(title)

        layout.addWidget(QLabel("DTE:"))
        self.dte = QSpinBox(); self.dte.setRange(1, 730); self.dte.setValue(int(dte))
        self.dte.valueChanged.connect(self.changed.emit)
        layout.addWidget(self.dte)

        self.atm_chk = QCheckBox("ATM")
        layout.addWidget(self.atm_chk)
        layout.addWidget(QLabel("|Δ|:"))
        self.delta = QDoubleSpinBox()
        self.delta.setRange(0.01, 0.50); self.delta.setSingleStep(0.05)
        self.delta.setDecimals(2); self.delta.setValue(
            float(delta_target) if delta_target is not None else 0.50)
        self.delta.valueChanged.connect(self.changed.emit)
        layout.addWidget(self.delta)

        if delta_target is None:
            self.atm_chk.setChecked(True)
            self.delta.setEnabled(False)
        self.atm_chk.stateChanged.connect(self._on_atm_toggle)
        layout.addStretch(1)

    def _on_atm_toggle(self, state):
        self.delta.setEnabled(state != Qt.Checked)
        self.changed.emit()

    def to_spec_kwargs(self) -> dict:
        return {
            "right": self.right, "side": self.side,
            "dte": self.dte.value(),
            "delta_target": None if self.atm_chk.isChecked() else float(self.delta.value()),
        }


class BacktestTab(QWidget):
    backtest_requested = pyqtSignal(str, object)   # ticker, BacktestConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._library = SignalLibrary()
        self._ticker: str | None = None
        self._last_result = None

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ---- Toolbar ----
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Start:"))
        self.start_date = QDateEdit(calendarPopup=True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.start_date.setDate(QDate.currentDate().addYears(-2))
        bar.addWidget(self.start_date)
        bar.addWidget(QLabel("End:"))
        self.end_date = QDateEdit(calendarPopup=True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setDate(QDate.currentDate())
        bar.addWidget(self.end_date)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Structure:"))
        self.structure_combo = QComboBox()
        for label, kind in STRUCTURES:
            self.structure_combo.addItem(label, kind)
        self.structure_combo.currentIndexChanged.connect(self._rebuild_leg_rows)
        bar.addWidget(self.structure_combo)
        bar.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["short", "long"])
        bar.addWidget(self.direction_combo)
        bar.addWidget(QLabel("Qty:"))
        self.qty_spin = QSpinBox(); self.qty_spin.setRange(1, 10000); self.qty_spin.setValue(1)
        bar.addWidget(self.qty_spin)

        bar.addSpacing(12)
        self.run_btn = QPushButton("Run backtest")
        self.run_btn.clicked.connect(self._on_run)
        bar.addWidget(self.run_btn)
        self.save_btn = QPushButton("Save config…")
        self.save_btn.clicked.connect(self._on_save_config)
        bar.addWidget(self.save_btn)
        self.load_btn = QPushButton("Load config…")
        self.load_btn.clicked.connect(self._on_load_config)
        bar.addWidget(self.load_btn)
        bar.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #bbb;")
        bar.addWidget(self.status_label)
        root.addLayout(bar)

        # ---- Leg specs ----
        self._legs_box = QGroupBox("Legs")
        self._legs_layout = QVBoxLayout(self._legs_box)
        self._legs_layout.setContentsMargins(6, 6, 6, 6)
        self._legs_layout.setSpacing(2)
        root.addWidget(self._legs_box)
        self._leg_rows: list[_LegSpecRow] = []

        # ---- Exits + hedge ----
        ex_h = QHBoxLayout()

        ex_box = QGroupBox("Structural exits")
        ex_layout = QHBoxLayout(ex_box)
        self.cb_dte_exit = QCheckBox("Close at DTE ≤")
        self.sp_dte_exit = QSpinBox(); self.sp_dte_exit.setRange(0, 365); self.sp_dte_exit.setValue(7)
        self.cb_profit = QCheckBox("Profit %")
        self.sp_profit = QDoubleSpinBox(); self.sp_profit.setRange(1, 1000); self.sp_profit.setValue(50.0)
        self.cb_stop = QCheckBox("Stop %")
        self.sp_stop = QDoubleSpinBox(); self.sp_stop.setRange(1, 5000); self.sp_stop.setValue(200.0)
        ex_layout.addWidget(self.cb_dte_exit); ex_layout.addWidget(self.sp_dte_exit)
        ex_layout.addWidget(self.cb_profit); ex_layout.addWidget(self.sp_profit)
        ex_layout.addWidget(self.cb_stop); ex_layout.addWidget(self.sp_stop)
        ex_layout.addWidget(QLabel("Re-arm:"))
        self.rearm_combo = QComboBox()
        self.rearm_combo.addItem("Any bar", "any_bar")
        self.rearm_combo.addItem("Edge only", "edge_only")
        ex_layout.addWidget(self.rearm_combo)
        ex_layout.addStretch(1)
        ex_h.addWidget(ex_box, 2)

        h_box = QGroupBox("Delta hedge")
        h_layout = QHBoxLayout(h_box)
        self.cb_h_int = QCheckBox("Every")
        self.sp_h_int = QSpinBox(); self.sp_h_int.setRange(1, 60); self.sp_h_int.setValue(5)
        self.cb_h_delta = QCheckBox("|Δ| >")
        self.sp_h_delta = QDoubleSpinBox(); self.sp_h_delta.setRange(1, 100000); self.sp_h_delta.setValue(50.0)
        self.cb_h_spot = QCheckBox("Spot move >")
        self.sp_h_spot = QDoubleSpinBox(); self.sp_h_spot.setRange(0.1, 50); self.sp_h_spot.setValue(2.0)
        h_layout.addWidget(self.cb_h_int); h_layout.addWidget(self.sp_h_int); h_layout.addWidget(QLabel("d"))
        h_layout.addWidget(self.cb_h_delta); h_layout.addWidget(self.sp_h_delta)
        h_layout.addWidget(self.cb_h_spot); h_layout.addWidget(self.sp_h_spot); h_layout.addWidget(QLabel("%"))
        h_layout.addStretch(1)
        ex_h.addWidget(h_box, 1)
        root.addLayout(ex_h)

        # ---- Rules + signal preview chart ----
        split = QSplitter(Qt.Horizontal)

        rules_panel = QWidget()
        rules_layout = QVBoxLayout(rules_panel)
        rules_layout.setContentsMargins(0, 0, 0, 0)
        self.entry_rule = RuleWidget(
            "Entry conditions", self._library, get_ticker=lambda: self._ticker)
        self.exit_rule = RuleWidget(
            "Exit conditions", self._library, get_ticker=lambda: self._ticker)
        rules_layout.addWidget(self.entry_rule)
        rules_layout.addWidget(self.exit_rule)
        rules_layout.addStretch(1)
        split.addWidget(rules_panel)

        chart_panel = QGroupBox("Signal preview")
        chart_layout = QVBoxLayout(chart_panel)
        self.signal_plot = pg.PlotWidget()
        self.signal_plot.setBackground("#111")
        self.signal_plot.showGrid(x=True, y=True, alpha=0.3)
        self.signal_plot.setLabel("bottom", "Date")
        self.signal_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        self.signal_plot.addLegend()
        chart_layout.addWidget(self.signal_plot)
        self.signal_hint = QLabel("Click a condition row to preview its signal.")
        self.signal_hint.setStyleSheet("color: #888;")
        chart_layout.addWidget(self.signal_hint)
        split.addWidget(chart_panel)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
        root.addWidget(split, 2)

        self.entry_rule.condition_selected.connect(
            lambda w: self._set_active_condition(w, role="entry"))
        self.exit_rule.condition_selected.connect(
            lambda w: self._set_active_condition(w, role="exit"))
        self.entry_rule.changed.connect(self._refresh_signal_preview)
        self.exit_rule.changed.connect(self._refresh_signal_preview)

        # ---- Equity curve + summary + trades ----
        self.pnl_plot = pg.PlotWidget()
        self.pnl_plot.setBackground("#111")
        self.pnl_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pnl_plot.setLabel("left", "Cum PnL ($)")
        self.pnl_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        self.pnl_curve = self.pnl_plot.plot(
            [], [], pen=pg.mkPen("#ffd166", width=2), name="Equity")
        self.pnl_plot.addItem(pg.InfiniteLine(pos=0, angle=0,
                              pen=pg.mkPen("#555", style=Qt.DashLine)))
        root.addWidget(self.pnl_plot, 2)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #ddd; padding: 4px;")
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        self.trades_table = QTableWidget(0, 8)
        self.trades_table.setHorizontalHeaderLabels(
            ["Entry", "Exit", "Days", "Strikes", "Sides/Rights",
             "Credit", "PnL", "Reason"])
        self.trades_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.trades_table.setMaximumHeight(180)
        root.addWidget(self.trades_table)

        # active condition tracking for signal preview
        self._active_cond_widget = None
        self._active_cond_role = None

        # Plot items used to render markers (created lazily)
        self._sig_curve = None
        self._sig_threshold_line = None
        self._sig_entry_scatter = None
        self._sig_exit_scatter = None
        self._sig_hedge_scatter = None

        # initial leg rows
        self._rebuild_leg_rows()

    # ------------------------------------------------------------------
    # Public API

    def set_ticker(self, ticker: str):
        self._ticker = ticker.upper()
        self._refresh_signal_preview()

    # ------------------------------------------------------------------
    # Leg row management

    def _current_kind(self) -> StructureKind:
        return self.structure_combo.currentData()

    def _rebuild_leg_rows(self):
        # Clear existing
        for row in self._leg_rows:
            self._legs_layout.removeWidget(row)
            row.setParent(None); row.deleteLater()
        self._leg_rows = []

        kind = self._current_kind()
        defaults = default_legs(kind)
        for i, spec in enumerate(defaults):
            row = _LegSpecRow(i, spec.right, spec.side, spec.dte, spec.delta_target)
            self._legs_layout.addWidget(row)
            self._leg_rows.append(row)

    def _build_structure_params(self) -> StructureParams:
        kind = self._current_kind()
        legs = []
        from ...analytics.structures import LegSpec
        for row in self._leg_rows:
            legs.append(LegSpec(**row.to_spec_kwargs()))
        return StructureParams(
            kind=kind,
            direction=self.direction_combo.currentText(),
            qty=self.qty_spin.value(),
            legs=legs,
        )

    # ------------------------------------------------------------------
    # Config build / apply

    def _build_config(self) -> BacktestConfig:
        hedge = HedgeConfig(
            use_interval=self.cb_h_int.isChecked(),
            interval_days=self.sp_h_int.value(),
            use_delta_threshold=self.cb_h_delta.isChecked(),
            delta_threshold=self.sp_h_delta.value(),
            use_spot_move=self.cb_h_spot.isChecked(),
            spot_move_pct=self.sp_h_spot.value(),
        )
        return BacktestConfig(
            start=self.start_date.date().toPyDate(),
            end=self.end_date.date().toPyDate(),
            structure=self._build_structure_params(),
            entry_rule=self.entry_rule.to_config(),
            exit_rule=self.exit_rule.to_config(),
            use_dte_exit=self.cb_dte_exit.isChecked(),
            dte_exit_threshold=self.sp_dte_exit.value(),
            use_profit_target=self.cb_profit.isChecked(),
            profit_target_pct=self.sp_profit.value(),
            use_stop_loss=self.cb_stop.isChecked(),
            stop_loss_pct=self.sp_stop.value(),
            rearm=self.rearm_combo.currentData(),
            hedge=hedge,
        )

    def _apply_config(self, cfg: BacktestConfig) -> None:
        if cfg.start:
            self.start_date.setDate(QDate(cfg.start.year, cfg.start.month, cfg.start.day))
        if cfg.end:
            self.end_date.setDate(QDate(cfg.end.year, cfg.end.month, cfg.end.day))

        # Structure
        idx = self.structure_combo.findData(cfg.structure.kind)
        if idx >= 0:
            self.structure_combo.setCurrentIndex(idx)
        self.direction_combo.setCurrentText(cfg.structure.direction)
        self.qty_spin.setValue(cfg.structure.qty)

        # Repopulate leg rows from the stored spec exactly
        for row in self._leg_rows:
            self._legs_layout.removeWidget(row); row.setParent(None); row.deleteLater()
        self._leg_rows = []
        for i, spec in enumerate(cfg.structure.legs):
            row = _LegSpecRow(i, spec.right, spec.side, spec.dte, spec.delta_target)
            self._legs_layout.addWidget(row)
            self._leg_rows.append(row)

        self.cb_dte_exit.setChecked(cfg.use_dte_exit)
        self.sp_dte_exit.setValue(cfg.dte_exit_threshold)
        self.cb_profit.setChecked(cfg.use_profit_target)
        self.sp_profit.setValue(cfg.profit_target_pct)
        self.cb_stop.setChecked(cfg.use_stop_loss)
        self.sp_stop.setValue(cfg.stop_loss_pct)
        ridx = self.rearm_combo.findData(cfg.rearm)
        if ridx >= 0:
            self.rearm_combo.setCurrentIndex(ridx)

        h = cfg.hedge
        self.cb_h_int.setChecked(h.use_interval); self.sp_h_int.setValue(h.interval_days)
        self.cb_h_delta.setChecked(h.use_delta_threshold); self.sp_h_delta.setValue(h.delta_threshold)
        self.cb_h_spot.setChecked(h.use_spot_move); self.sp_h_spot.setValue(h.spot_move_pct)

        self.entry_rule.set_from_config(cfg.entry_rule)
        self.exit_rule.set_from_config(cfg.exit_rule)

    # ------------------------------------------------------------------
    # Signal preview chart

    def _set_active_condition(self, w, role: str):
        self._active_cond_widget = w
        self._active_cond_role = role
        self._refresh_signal_preview()

    def _refresh_signal_preview(self):
        # Clear plot
        self.signal_plot.clear()
        self.signal_plot.addLegend()
        self._sig_curve = None
        self._sig_threshold_line = None
        self._sig_entry_scatter = None
        self._sig_exit_scatter = None
        self._sig_hedge_scatter = None

        if self._active_cond_widget is None or self._ticker is None:
            self.signal_hint.setText(
                "Load a ticker and click a condition row to preview its signal.")
            return

        # Build the signal from the condition's current dict
        cfg = self._active_cond_widget.to_config()
        if cfg is None:
            self.signal_hint.setText("Condition has no signal selected.")
            return
        try:
            sig = signal_from_dict(cfg.signal)
            series = sig.series(self._ticker)
        except Exception as e:
            self.signal_hint.setText(f"Could not compute signal: {e}")
            return
        if series is None or series.empty:
            self.signal_hint.setText("Signal series is empty for this ticker.")
            return

        idx = pd.to_datetime(series.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        xs = np.asarray(idx.view("int64") // 1_000_000_000)
        ys = series.astype(float).to_numpy()
        valid = np.isfinite(ys)
        if not valid.any():
            self.signal_hint.setText("Signal series has no valid points.")
            return
        xs_v = xs[valid]
        ys_v = ys[valid]

        name = self._active_cond_widget.selected_signal_name()
        self._sig_curve = self.signal_plot.plot(
            xs_v, ys_v, pen=pg.mkPen("#4aa3ff", width=2), name=name)
        # Threshold line
        self._sig_threshold_line = pg.InfiniteLine(
            pos=cfg.threshold, angle=0,
            pen=pg.mkPen("#ffd166", width=1, style=Qt.DashLine))
        self.signal_plot.addItem(self._sig_threshold_line)

        self.signal_plot.setLabel("left", name)
        self.signal_hint.setText(
            f"{name} • {len(xs_v):,} points • threshold = {cfg.threshold:g}"
            + (f" • role: {self._active_cond_role}" if self._active_cond_role else "")
        )

        self._overlay_trade_markers(series)

    def _overlay_trade_markers(self, series: pd.Series):
        """If a backtest result is available, overlay entry/exit/hedge markers
        sampled at the matching dates of the active signal series.
        """
        if self._last_result is None:
            return
        res = self._last_result
        if res.error:
            return

        idx = pd.to_datetime(series.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        # quick lookup: date -> y-value
        ymap = pd.Series(series.astype(float).values, index=idx).to_dict()

        def _markers_for(dates: list[date]):
            xs, ys = [], []
            for d in dates:
                ts = pd.Timestamp(d).normalize()
                v = ymap.get(ts)
                if v is None or not np.isfinite(v):
                    continue
                xs.append(int(ts.value // 1_000_000_000))
                ys.append(float(v))
            return xs, ys

        if res.trades:
            ex_dates = [t.entry_date for t in res.trades]
            xs_e, ys_e = _markers_for(ex_dates)
            if xs_e:
                sc_e = pg.ScatterPlotItem(
                    xs_e, ys_e, symbol="t1", size=12,
                    pen=pg.mkPen("#0a0", width=1), brush=pg.mkBrush("#3f3"))
                self.signal_plot.addItem(sc_e)
                self._sig_entry_scatter = sc_e

            xs_x, ys_x = _markers_for([t.exit_date for t in res.trades])
            if xs_x:
                sc_x = pg.ScatterPlotItem(
                    xs_x, ys_x, symbol="t", size=12,
                    pen=pg.mkPen("#a00", width=1), brush=pg.mkBrush("#f55"))
                self.signal_plot.addItem(sc_x)
                self._sig_exit_scatter = sc_x

        if res.hedge_events:
            xs_h, ys_h = _markers_for([h.date for h in res.hedge_events])
            if xs_h:
                sc_h = pg.ScatterPlotItem(
                    xs_h, ys_h, symbol="d", size=10,
                    pen=pg.mkPen("#fa0", width=1), brush=pg.mkBrush("#fc6"))
                self.signal_plot.addItem(sc_h)
                self._sig_hedge_scatter = sc_h

    # ------------------------------------------------------------------
    # Run + result wiring

    def _on_run(self):
        if not self._ticker:
            self.status_label.setText("Load a ticker first.")
            return
        try:
            cfg = self._build_config()
        except Exception as e:
            self.status_label.setText(f"Config error: {e}")
            return
        if cfg.entry_rule.is_empty():
            self.status_label.setText(
                "Add at least one entry condition before running.")
            return
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running…")
        self.backtest_requested.emit(self._ticker, cfg)

    def on_backtest_progress(self, cur: int, total: int, msg: str):
        self.status_label.setText(f"{msg}  ({cur}/{total})")

    def on_backtest_result(self, result):
        self.run_btn.setEnabled(True)
        self._last_result = result
        if result is None:
            self.status_label.setText("No result.")
            return
        if result.error:
            self.status_label.setText(f"Error: {result.error}")
            self.pnl_curve.setData([], [])
            self.summary_label.setText("")
            self.trades_table.setRowCount(0)
            return
        eq = result.equity_curve
        if eq.empty:
            self.status_label.setText("No equity curve produced.")
            self.pnl_curve.setData([], [])
            self.summary_label.setText("")
            self.trades_table.setRowCount(0)
            return

        xs = pd.to_datetime(eq["date"]).view("int64") // 1_000_000_000
        self.pnl_curve.setData(xs.to_numpy(), eq["cum_pnl"].to_numpy())

        s = result.summary
        self.summary_label.setText(
            f"Total PnL: ${s.get('total_pnl', 0):,.0f}  |  "
            f"Trades: {s.get('n_trades', 0)}  |  "
            f"Win rate: {s.get('win_rate', 0) * 100:.1f}%  |  "
            f"Avg trade: ${s.get('avg_trade_pnl', 0):,.0f}  |  "
            f"Avg hold: {s.get('avg_hold_days', 0):.1f}d  |  "
            f"Max DD: ${s.get('max_drawdown', 0):,.0f}  |  "
            f"Sharpe: {s.get('sharpe', 0):.2f}  |  "
            f"Hedges: {s.get('n_hedges', 0)}"
        )
        self.status_label.setText(
            f"Done. {s.get('n_trades', 0)} trades, "
            f"{s.get('n_hedges', 0)} hedges.")

        self._populate_trades(result.trades)
        # refresh signal preview to overlay markers
        self._refresh_signal_preview()

    def on_backtest_failed(self, msg: str):
        self.run_btn.setEnabled(True)
        first = msg.splitlines()[0] if msg else ""
        self.status_label.setText(f"Failed: {first}")

    def _populate_trades(self, trades):
        self.trades_table.setRowCount(len(trades))
        for r, t in enumerate(trades):
            days = (t.exit_date - t.entry_date).days
            sides_rights = ", ".join(f"{s[0].upper()}{rt}" for s, rt in zip(t.sides, t.rights))
            cells = [
                str(t.entry_date), str(t.exit_date), str(days),
                "/".join(f"{k:g}" for k in t.strikes),
                sides_rights,
                f"${t.credit:,.2f}", f"${t.pnl:,.2f}", t.exit_reason,
            ]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if c == 6:  # PnL
                    if t.pnl >= 0:
                        item.setForeground(pg.mkBrush("#5fcf5f").color())
                    else:
                        item.setForeground(pg.mkBrush("#ef6262").color())
                self.trades_table.setItem(r, c, item)

    # ------------------------------------------------------------------
    # Save / load configs (QSettings, JSON list of {name, config_dict})

    def _saved_configs(self) -> list[dict]:
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        raw = s.value(CONFIGS_KEY, "", type=str)
        if not raw:
            return []
        try:
            return json.loads(raw) or []
        except Exception:
            return []

    def _write_saved_configs(self, items: list[dict]) -> None:
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        s.setValue(CONFIGS_KEY, json.dumps(items))

    def _on_save_config(self):
        try:
            cfg = self._build_config()
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Config error: {e}")
            return
        name, ok = QInputDialog.getText(self, "Save backtest config", "Name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        items = self._saved_configs()
        items = [it for it in items if it.get("name") != name]
        items.append({"name": name, "config": cfg.to_dict()})
        self._write_saved_configs(items)
        self.status_label.setText(f"Saved config '{name}'.")

    def _on_load_config(self):
        items = self._saved_configs()
        if not items:
            QMessageBox.information(self, "No configs", "No saved configs yet.")
            return
        names = [it["name"] for it in items]
        name, ok = QInputDialog.getItem(
            self, "Load backtest config", "Choose:", names, 0, False)
        if not ok:
            return
        chosen = next((it for it in items if it.get("name") == name), None)
        if chosen is None:
            return
        try:
            cfg = BacktestConfig.from_dict(chosen["config"])
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self._apply_config(cfg)
        self.status_label.setText(f"Loaded config '{name}'.")
