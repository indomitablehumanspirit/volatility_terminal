"""Backtest tab — composable signal-driven backtests over historical chains.

Layout (top to bottom):
  - Toolbar: date range, "Add leg", run, save/load.
  - Leg-spec rows: free-form, user adds/removes; each row has right/side/DTE/Δ/qty.
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
from PyQt5.QtCore import QDate, QSettings, Qt, QThreadPool, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...analytics.backtest_config import BacktestConfig
from ...analytics.signals import (
    Condition, RuleConfig, signal_from_dict,
)
from ...analytics.simulation import HedgeConfig
from ...analytics.structures import LegSpec, StructureParams
from ...analytics.tuning import TuningConfig, TuningParam
from ..rule_widget import RuleWidget
from ..signal_library import SignalLibrary
from ..workers import Worker
from .tuning_results_widget import TuningResultsWidget


SETTINGS_ORG = "VolatilityTerminal"
SETTINGS_APP = "VolatilityTerminal"
CONFIGS_KEY = "backtest/saved_configs"


class _LegSpecRow(QWidget):
    changed = pyqtSignal()
    removed = pyqtSignal(object)   # emits self

    def __init__(self, right: str, side: str, dte: int,
                 delta_target: float | None, qty: int = 1, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.title = QLabel("Leg ?")
        self.title.setMinimumWidth(50)
        layout.addWidget(self.title)

        self.right_combo = QComboBox()
        self.right_combo.addItems(["C", "P"])
        self.right_combo.setCurrentText(right)
        self.right_combo.currentIndexChanged.connect(self.changed.emit)
        layout.addWidget(self.right_combo)

        self.side_combo = QComboBox()
        self.side_combo.addItems(["short", "long"])
        self.side_combo.setCurrentText(side)
        self.side_combo.currentIndexChanged.connect(self.changed.emit)
        layout.addWidget(self.side_combo)

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

        layout.addWidget(QLabel("Qty:"))
        self.qty_spin = QSpinBox(); self.qty_spin.setRange(1, 10000); self.qty_spin.setValue(int(qty))
        self.qty_spin.valueChanged.connect(self.changed.emit)
        layout.addWidget(self.qty_spin)

        self.remove_btn = QPushButton("✕")
        self.remove_btn.setMaximumWidth(28)
        self.remove_btn.setToolTip("Remove this leg")
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.remove_btn)
        layout.addStretch(1)

    def _on_atm_toggle(self, state):
        self.delta.setEnabled(state != Qt.Checked)
        self.changed.emit()

    def set_index(self, idx: int) -> None:
        self.title.setText(f"Leg {idx+1}")

    def to_spec_kwargs(self) -> dict:
        return {
            "right": self.right_combo.currentText(),
            "side": self.side_combo.currentText(),
            "dte": self.dte.value(),
            "delta_target": None if self.atm_chk.isChecked() else float(self.delta.value()),
            "qty": int(self.qty_spin.value()),
        }


class BacktestTab(QWidget):
    backtest_requested = pyqtSignal(str, object)   # ticker, BacktestConfig
    tune_requested = pyqtSignal(str, object, object)  # ticker, BacktestConfig, TuningConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._library = SignalLibrary()
        self._ticker: str | None = None
        self._last_result = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        self._content = QWidget()
        scroll.setWidget(self._content)
        root = QVBoxLayout(self._content)
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
        self.add_leg_btn = QPushButton("Add leg")
        self.add_leg_btn.clicked.connect(self._on_add_leg)
        bar.addWidget(self.add_leg_btn)

        bar.addSpacing(12)
        self.run_btn = QPushButton("Run backtest")
        self.run_btn.clicked.connect(self._on_run)
        bar.addWidget(self.run_btn)
        self.tune_btn = QPushButton("Tune…")
        self.tune_btn.clicked.connect(self._on_tune)
        bar.addWidget(self.tune_btn)
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

        # ---- Tuning controls (hidden until "Tune…" clicked) ----
        self._tune_group = QGroupBox("Hyperparameter tuning")
        tune_root = QVBoxLayout(self._tune_group)
        tune_settings = QHBoxLayout()
        tune_settings.addWidget(QLabel("IS split:"))
        self.tune_split_spin = QSpinBox()
        self.tune_split_spin.setRange(50, 90); self.tune_split_spin.setValue(70)
        self.tune_split_spin.setSuffix(" %")
        tune_settings.addWidget(self.tune_split_spin)
        tune_settings.addWidget(QLabel("Trials:"))
        self.tune_trials_spin = QSpinBox()
        self.tune_trials_spin.setRange(50, 2000); self.tune_trials_spin.setValue(300)
        tune_settings.addWidget(self.tune_trials_spin)
        tune_settings.addWidget(QLabel("Objective:"))
        self.tune_obj_combo = QComboBox()
        for label, val in [("Profit factor", "profit_factor"),
                            ("Sharpe ratio", "sharpe"),
                            ("Total PnL ⚠", "total_pnl")]:
            self.tune_obj_combo.addItem(label, val)
        tune_settings.addWidget(self.tune_obj_combo)
        self.tune_obj_warn = QLabel(
            "Note: Total PnL has highest overfitting bias — prefer Profit factor.")
        self.tune_obj_warn.setStyleSheet("color: #888;")
        self.tune_obj_warn.setVisible(False)
        self.tune_obj_combo.currentIndexChanged.connect(
            lambda _: self.tune_obj_warn.setVisible(
                self.tune_obj_combo.currentData() == "total_pnl"))
        tune_settings.addWidget(self.tune_obj_warn)
        tune_settings.addWidget(QLabel("Min trades:"))
        self.tune_min_trades_spin = QSpinBox()
        self.tune_min_trades_spin.setRange(1, 100)
        self.tune_min_trades_spin.setValue(5)
        tune_settings.addWidget(self.tune_min_trades_spin)
        tune_settings.addStretch(1)
        tune_root.addLayout(tune_settings)

        self.tune_params_table = QTableWidget(0, 4)
        self.tune_params_table.setHorizontalHeaderLabels(
            ["Tune?", "Parameter", "Min", "Max"])
        self.tune_params_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.tune_params_table.verticalHeader().setVisible(False)
        self.tune_params_table.setMaximumHeight(220)
        self.tune_params_table.setMinimumHeight(140)
        tune_root.addWidget(self.tune_params_table)

        tune_action = QHBoxLayout()
        self.tune_run_btn = QPushButton("Run tuning")
        self.tune_run_btn.clicked.connect(self._on_tune_run)
        tune_action.addWidget(self.tune_run_btn)
        self.tune_status_label = QLabel("")
        self.tune_status_label.setStyleSheet("color: #bbb;")
        tune_action.addWidget(self.tune_status_label)
        tune_action.addStretch(1)
        tune_root.addLayout(tune_action)

        self._tune_group.setVisible(False)
        root.addWidget(self._tune_group)

        # Tracks the currently-displayed results widget (cleared on each run)
        self._tuning_widget: TuningResultsWidget | None = None
        # Internal name → display label, captured at table-build time so
        # results can be labeled identically.
        self._tune_param_labels: dict[str, str] = {}
        # Internal name → TuningParam metadata for the param table rows
        self._tune_table_rows: list[tuple[TuningParam, "QCheckBox",
                                           "QDoubleSpinBox", "QDoubleSpinBox"]] = []

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
        self._sig_worker = None
        self._sig_generation = 0

        # initial leg rows: one default short ATM call at 30 DTE
        self._add_leg_row(LegSpec(right="C", side="short", dte=30, delta_target=None, qty=1))
        self._renumber_leg_rows()
        self._rebuild_tune_params_table()

    # ------------------------------------------------------------------
    # Public API

    def set_ticker(self, ticker: str):
        self._ticker = ticker.upper()
        self._refresh_signal_preview()

    # ------------------------------------------------------------------
    # Leg row management

    def _add_leg_row(self, spec: LegSpec) -> _LegSpecRow:
        row = _LegSpecRow(spec.right, spec.side, spec.dte, spec.delta_target, spec.qty)
        row.removed.connect(self._on_remove_leg)
        self._legs_layout.addWidget(row)
        self._leg_rows.append(row)
        return row

    def _renumber_leg_rows(self) -> None:
        for i, row in enumerate(self._leg_rows):
            row.set_index(i)

    def _clear_leg_rows(self) -> None:
        for row in self._leg_rows:
            self._legs_layout.removeWidget(row)
            row.setParent(None); row.deleteLater()
        self._leg_rows = []

    def _on_add_leg(self) -> None:
        self._add_leg_row(LegSpec(right="C", side="short", dte=30, delta_target=None, qty=1))
        self._renumber_leg_rows()
        self._rebuild_tune_params_table()

    def _on_remove_leg(self, row: _LegSpecRow) -> None:
        if row in self._leg_rows:
            self._leg_rows.remove(row)
        self._legs_layout.removeWidget(row)
        row.setParent(None); row.deleteLater()
        self._renumber_leg_rows()
        self._rebuild_tune_params_table()

    def _build_structure_params(self) -> StructureParams:
        legs = [LegSpec(**row.to_spec_kwargs()) for row in self._leg_rows]
        return StructureParams(legs=legs)

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

        # Repopulate leg rows from the stored spec exactly
        self._clear_leg_rows()
        for spec in cfg.structure.legs:
            self._add_leg_row(spec)
        self._renumber_leg_rows()

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

        self._rebuild_tune_params_table()

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

        cfg = self._active_cond_widget.to_config()
        if cfg is None:
            self.signal_hint.setText("Condition has no signal selected.")
            return

        self._sig_generation += 1
        gen = self._sig_generation
        sig_dict = cfg.signal
        ticker = self._ticker
        threshold = cfg.threshold

        self.signal_hint.setText("Computing signal…")

        def _compute():
            sig = signal_from_dict(sig_dict)
            return sig.series(ticker)

        worker = Worker(_compute)
        worker.signals.finished.connect(
            lambda series, _g=gen, _t=threshold: self._on_signal_ready(
                series, _g, _t))
        worker.signals.failed.connect(
            lambda msg, _g=gen: self._on_signal_failed(msg, _g))
        self._sig_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_signal_failed(self, msg: str, generation: int):
        if generation != self._sig_generation:
            return
        self.signal_hint.setText(f"Could not compute signal: {msg}")

    def _on_signal_ready(self, series, generation: int, threshold: float):
        if generation != self._sig_generation:
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

        name = (self._active_cond_widget.selected_signal_name()
                if self._active_cond_widget else "Signal")
        self._sig_curve = self.signal_plot.plot(
            xs_v, ys_v, pen=pg.mkPen("#4aa3ff", width=2), name=name)
        self._sig_threshold_line = pg.InfiniteLine(
            pos=threshold, angle=0,
            pen=pg.mkPen("#ffd166", width=1, style=Qt.DashLine))
        self.signal_plot.addItem(self._sig_threshold_line)

        self.signal_plot.setLabel("left", name)
        self.signal_hint.setText(
            f"{name} • {len(xs_v):,} points • threshold = {threshold:g}"
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
    # Hyperparameter tuning

    def _candidate_tune_params(self) -> list[tuple[TuningParam, float]]:
        """Build the candidate parameter rows (param + current value) from the
        current UI state. Returned in display order."""
        items: list[tuple[TuningParam, float]] = []
        # Per-leg DTE
        for i, row in enumerate(self._leg_rows):
            cur = int(row.dte.value())
            items.append((TuningParam(
                name=f"leg_{i}_dte", label=f"Leg {i+1} DTE",
                min_val=float(max(1, cur - 14)),
                max_val=float(cur + 21), is_integer=True), float(cur)))
        # Per-leg |Δ| (only if not ATM)
        for i, row in enumerate(self._leg_rows):
            if row.atm_chk.isChecked():
                continue
            cur = float(row.delta.value())
            items.append((TuningParam(
                name=f"leg_{i}_delta", label=f"Leg {i+1} |Δ|",
                min_val=max(0.05, cur - 0.15),
                max_val=min(0.50, cur + 0.15), is_integer=False), cur))
        # Structural exits
        if self.cb_profit.isChecked():
            items.append((TuningParam(
                name="profit_target_pct", label="Profit target %",
                min_val=20.0, max_val=100.0, is_integer=False),
                float(self.sp_profit.value())))
        if self.cb_stop.isChecked():
            items.append((TuningParam(
                name="stop_loss_pct", label="Stop loss %",
                min_val=50.0, max_val=400.0, is_integer=False),
                float(self.sp_stop.value())))
        if self.cb_dte_exit.isChecked():
            items.append((TuningParam(
                name="dte_exit_threshold", label="DTE exit threshold",
                min_val=0.0, max_val=30.0, is_integer=True),
                float(self.sp_dte_exit.value())))
        # Entry / exit condition thresholds
        for i, c in enumerate(self.entry_rule.conditions()):
            cfg = c.to_config()
            if cfg is None:
                continue
            cur = float(cfg.threshold)
            mn = cur * 0.5 if cur >= 0 else cur * 1.5
            mx = cur * 1.5 if cur >= 0 else cur * 0.5
            if mn == mx:
                mn, mx = cur - 1.0, cur + 1.0
            items.append((TuningParam(
                name=f"entry_cond_{i}_threshold",
                label=f"Entry cond {i+1} threshold",
                min_val=min(mn, mx), max_val=max(mn, mx),
                is_integer=False), cur))
        for i, c in enumerate(self.exit_rule.conditions()):
            cfg = c.to_config()
            if cfg is None:
                continue
            cur = float(cfg.threshold)
            mn = cur * 0.5 if cur >= 0 else cur * 1.5
            mx = cur * 1.5 if cur >= 0 else cur * 0.5
            if mn == mx:
                mn, mx = cur - 1.0, cur + 1.0
            items.append((TuningParam(
                name=f"exit_cond_{i}_threshold",
                label=f"Exit cond {i+1} threshold",
                min_val=min(mn, mx), max_val=max(mn, mx),
                is_integer=False), cur))
        # Hedge
        if self.cb_h_int.isChecked():
            items.append((TuningParam(
                name="hedge_interval_days", label="Hedge interval (days)",
                min_val=1.0, max_val=20.0, is_integer=True),
                float(self.sp_h_int.value())))
        if self.cb_h_delta.isChecked():
            items.append((TuningParam(
                name="hedge_delta_threshold", label="Hedge |Δ| threshold",
                min_val=10.0, max_val=200.0, is_integer=False),
                float(self.sp_h_delta.value())))
        if self.cb_h_spot.isChecked():
            items.append((TuningParam(
                name="hedge_spot_move_pct", label="Hedge spot move %",
                min_val=0.5, max_val=10.0, is_integer=False),
                float(self.sp_h_spot.value())))
        return items

    def _rebuild_tune_params_table(self):
        """Repopulate the params table from the current backtest config."""
        # Capture prior selection state to preserve user check choices on rebuild
        prior_checked: dict[str, tuple[float, float]] = {}
        for p, chk, mn, mx in self._tune_table_rows:
            if chk.isChecked():
                prior_checked[p.name] = (mn.value(), mx.value())

        cands = self._candidate_tune_params()
        self._tune_table_rows = []
        self._tune_param_labels = {p.name: p.label for p, _ in cands}
        self.tune_params_table.setRowCount(len(cands))
        for r, (p, _cur) in enumerate(cands):
            chk = QCheckBox()
            container = QWidget()
            ch_lay = QHBoxLayout(container)
            ch_lay.setContentsMargins(6, 0, 0, 0)
            ch_lay.addWidget(chk)
            ch_lay.addStretch(1)
            self.tune_params_table.setCellWidget(r, 0, container)

            self.tune_params_table.setItem(r, 1, QTableWidgetItem(p.label))
            self.tune_params_table.item(r, 1).setFlags(Qt.ItemIsEnabled)

            def _mk_spin(default: float, integer: bool) -> QDoubleSpinBox:
                sp = QDoubleSpinBox()
                sp.setDecimals(0 if integer else 4)
                sp.setRange(-1e9, 1e9)
                sp.setSingleStep(1.0 if integer else 0.05)
                sp.setValue(float(default))
                return sp

            mn = _mk_spin(p.min_val, p.is_integer)
            mx = _mk_spin(p.max_val, p.is_integer)
            self.tune_params_table.setCellWidget(r, 2, mn)
            self.tune_params_table.setCellWidget(r, 3, mx)

            if p.name in prior_checked:
                chk.setChecked(True)
                pmn, pmx = prior_checked[p.name]
                mn.setValue(pmn)
                mx.setValue(pmx)

            self._tune_table_rows.append((p, chk, mn, mx))

    def _on_tune(self):
        # Toggle visibility, rebuild table to capture current state
        self._tune_group.setVisible(not self._tune_group.isVisible())
        if self._tune_group.isVisible():
            self._rebuild_tune_params_table()

    def _on_tune_run(self):
        if not self._ticker:
            self.tune_status_label.setText("Load a ticker first.")
            return
        try:
            base_cfg = self._build_config()
        except Exception as e:
            self.tune_status_label.setText(f"Config error: {e}")
            return
        if base_cfg.entry_rule.is_empty():
            self.tune_status_label.setText(
                "Add at least one entry condition before tuning.")
            return

        chosen: list[TuningParam] = []
        for p, chk, mn, mx in self._tune_table_rows:
            if not chk.isChecked():
                continue
            lo, hi = float(mn.value()), float(mx.value())
            if lo > hi:
                lo, hi = hi, lo
            chosen.append(TuningParam(
                name=p.name, label=p.label,
                min_val=lo, max_val=hi, is_integer=p.is_integer))
        if not chosen:
            self.tune_status_label.setText(
                "Check at least one parameter to tune.")
            return

        tcfg = TuningConfig(
            params=chosen,
            n_trials=int(self.tune_trials_spin.value()),
            split_pct=float(self.tune_split_spin.value()) / 100.0,
            objective=self.tune_obj_combo.currentData(),
            min_trades=int(self.tune_min_trades_spin.value()),
        )
        self.tune_run_btn.setEnabled(False)
        self.tune_status_label.setText("Tuning…")
        self.tune_requested.emit(self._ticker, base_cfg, tcfg)

    def on_tune_progress(self, cur: int, total: int, msg: str):
        self.tune_status_label.setText(f"{msg}  ({cur}/{total})")

    def on_tune_result(self, result):
        self.tune_run_btn.setEnabled(True)
        if result is None:
            self.tune_status_label.setText("No tuning result.")
            return
        if result.error:
            self.tune_status_label.setText(f"Tuning error: {result.error}")
            return

        # Replace any prior results widget
        if self._tuning_widget is not None:
            self._root_layout().removeWidget(self._tuning_widget)
            self._tuning_widget.setParent(None)
            self._tuning_widget.deleteLater()
            self._tuning_widget = None

        w = TuningResultsWidget(result)
        w.set_param_labels(self._tune_param_labels)
        w.params_apply_requested.connect(self._apply_tuned_params)

        # Insert immediately below the tuning group box
        layout = self._root_layout()
        idx = layout.indexOf(self._tune_group)
        layout.insertWidget(idx + 1, w)
        self._tuning_widget = w

        is_s = result.best_is_score
        oos_s = result.oos_score
        self.tune_status_label.setText(
            f"Tuning complete. Best IS: {is_s:.3f} | OOS: {oos_s:.3f}")

    def on_tune_failed(self, msg: str):
        self.tune_run_btn.setEnabled(True)
        first = msg.splitlines()[0] if msg else ""
        self.tune_status_label.setText(f"Tuning failed: {first}")

    def _root_layout(self) -> QVBoxLayout:
        return self._content.layout()  # type: ignore[return-value]

    def _apply_tuned_params(self, params: dict):
        for name, v in params.items():
            if name.startswith("leg_") and name.endswith("_dte"):
                i = int(name.split("_")[1])
                if 0 <= i < len(self._leg_rows):
                    self._leg_rows[i].dte.setValue(int(round(v)))
            elif name.startswith("leg_") and name.endswith("_delta"):
                i = int(name.split("_")[1])
                if 0 <= i < len(self._leg_rows):
                    if self._leg_rows[i].atm_chk.isChecked():
                        self._leg_rows[i].atm_chk.setChecked(False)
                    self._leg_rows[i].delta.setValue(float(v))
            elif name == "profit_target_pct":
                self.sp_profit.setValue(float(v))
            elif name == "stop_loss_pct":
                self.sp_stop.setValue(float(v))
            elif name == "dte_exit_threshold":
                self.sp_dte_exit.setValue(int(round(v)))
            elif name.startswith("entry_cond_") and name.endswith("_threshold"):
                i = int(name.split("_")[2])
                conds = self.entry_rule.conditions()
                if 0 <= i < len(conds):
                    conds[i].thr_spin.setValue(float(v))
            elif name.startswith("exit_cond_") and name.endswith("_threshold"):
                i = int(name.split("_")[2])
                conds = self.exit_rule.conditions()
                if 0 <= i < len(conds):
                    conds[i].thr_spin.setValue(float(v))
            elif name == "hedge_interval_days":
                self.sp_h_int.setValue(int(round(v)))
            elif name == "hedge_delta_threshold":
                self.sp_h_delta.setValue(float(v))
            elif name == "hedge_spot_move_pct":
                self.sp_h_spot.setValue(float(v))
        self.status_label.setText(
            "Best params applied — click Run backtest to see OOS equity curve.")

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
