"""VRP tab: ATM-IV timeseries at fixed DTE vs rolling realized vol.

Also hosts a rolling short-straddle / short-strangle backtest: a collapsible
panel below the VRP bar plot runs ``run_straddle_backtest`` off-thread and
renders the cumulative PnL curve plus a summary line.
"""
from __future__ import annotations

import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ...analytics.iv_timeseries import build_iv_timeseries
from ...analytics.realized import close_to_close
from ...analytics.simulation import HedgeConfig
from ...analytics.straddle_backtest import StraddleBacktestConfig
from ...analytics.vrp import compute_vrp
from ...data import cache


class VrpTab(QWidget):
    rebuild_requested = pyqtSignal(str, int)  # ticker, dte
    backtest_requested = pyqtSignal(str, object)  # ticker, StraddleBacktestConfig

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

        # --- Short-vol backtest panel (collapsible) ---
        self.bt_group = QGroupBox("Short-vol backtest")
        self.bt_group.setCheckable(True)
        self.bt_group.setChecked(False)
        self.bt_group.toggled.connect(self._on_bt_toggled)
        root.addWidget(self.bt_group)

        bt_root = QVBoxLayout(self.bt_group)
        self._bt_body = QWidget()
        bt_root.addWidget(self._bt_body)
        body = QGridLayout(self._bt_body)
        body.setContentsMargins(6, 6, 6, 6)

        # row 0: structure / DTE / qty / strangle delta
        body.addWidget(QLabel("Structure:"), 0, 0)
        self.structure_combo = QComboBox()
        self.structure_combo.addItems(["Straddle", "Strangle"])
        self.structure_combo.currentTextChanged.connect(self._on_structure_changed)
        body.addWidget(self.structure_combo, 0, 1)

        body.addWidget(QLabel("Target DTE:"), 0, 2)
        self.bt_dte = QSpinBox()
        self.bt_dte.setRange(1, 365)
        self.bt_dte.setValue(30)
        body.addWidget(self.bt_dte, 0, 3)

        body.addWidget(QLabel("Qty:"), 0, 4)
        self.bt_qty = QSpinBox()
        self.bt_qty.setRange(1, 1000)
        self.bt_qty.setValue(1)
        body.addWidget(self.bt_qty, 0, 5)

        body.addWidget(QLabel("|Δ| wings:"), 0, 6)
        self.bt_strangle_delta = QDoubleSpinBox()
        self.bt_strangle_delta.setRange(0.05, 0.45)
        self.bt_strangle_delta.setSingleStep(0.05)
        self.bt_strangle_delta.setValue(0.25)
        self.bt_strangle_delta.setEnabled(False)
        body.addWidget(self.bt_strangle_delta, 0, 7)

        # row 1: entry filters
        self.cb_ivrank = QCheckBox("IV rank >")
        body.addWidget(self.cb_ivrank, 1, 0)
        self.sp_ivrank_thr = QDoubleSpinBox()
        self.sp_ivrank_thr.setRange(0.0, 100.0); self.sp_ivrank_thr.setValue(50.0)
        body.addWidget(self.sp_ivrank_thr, 1, 1)
        body.addWidget(QLabel("Lookback:"), 1, 2)
        self.sp_ivrank_lb = QSpinBox()
        self.sp_ivrank_lb.setRange(20, 1260); self.sp_ivrank_lb.setValue(252)
        body.addWidget(self.sp_ivrank_lb, 1, 3)
        body.addWidget(QLabel("Method:"), 1, 4)
        self.cb_ivrank_method = QComboBox()
        self.cb_ivrank_method.addItems(["Rank", "Percentile"])
        body.addWidget(self.cb_ivrank_method, 1, 5)
        self.cb_vrp = QCheckBox("VRP >")
        body.addWidget(self.cb_vrp, 1, 6)
        self.sp_vrp_thr = QDoubleSpinBox()
        self.sp_vrp_thr.setRange(-1.0, 1.0); self.sp_vrp_thr.setDecimals(3)
        self.sp_vrp_thr.setSingleStep(0.01); self.sp_vrp_thr.setValue(0.02)
        body.addWidget(self.sp_vrp_thr, 1, 7)

        self.cb_conditional = QCheckBox("Conditional-only re-entry (stay flat until filter passes)")
        body.addWidget(self.cb_conditional, 2, 0, 1, 8)

        # row 3: exit rules
        self.cb_dte_exit = QCheckBox("Close at DTE ≤")
        body.addWidget(self.cb_dte_exit, 3, 0)
        self.sp_dte_exit = QSpinBox()
        self.sp_dte_exit.setRange(0, 365); self.sp_dte_exit.setValue(21)
        body.addWidget(self.sp_dte_exit, 3, 1)
        self.cb_profit = QCheckBox("Profit target %")
        body.addWidget(self.cb_profit, 3, 2)
        self.sp_profit = QDoubleSpinBox()
        self.sp_profit.setRange(1.0, 500.0); self.sp_profit.setValue(50.0)
        body.addWidget(self.sp_profit, 3, 3)
        self.cb_stop = QCheckBox("Stop loss %")
        body.addWidget(self.cb_stop, 3, 4)
        self.sp_stop = QDoubleSpinBox()
        self.sp_stop.setRange(1.0, 2000.0); self.sp_stop.setValue(200.0)
        body.addWidget(self.sp_stop, 3, 5)

        # row 4: hedging
        self.cb_h_int = QCheckBox("Hedge every")
        body.addWidget(self.cb_h_int, 4, 0)
        self.sp_h_int = QSpinBox()
        self.sp_h_int.setRange(1, 60); self.sp_h_int.setValue(5)
        body.addWidget(self.sp_h_int, 4, 1)
        self.cb_h_delta = QCheckBox("|Δ| >")
        body.addWidget(self.cb_h_delta, 4, 2)
        self.sp_h_delta = QDoubleSpinBox()
        self.sp_h_delta.setRange(1.0, 10000.0); self.sp_h_delta.setValue(50.0)
        body.addWidget(self.sp_h_delta, 4, 3)
        self.cb_h_spot = QCheckBox("Spot move >")
        body.addWidget(self.cb_h_spot, 4, 4)
        self.sp_h_spot = QDoubleSpinBox()
        self.sp_h_spot.setRange(0.1, 50.0); self.sp_h_spot.setValue(2.0)
        body.addWidget(self.sp_h_spot, 4, 5)

        # row 5: run + status
        self.run_btn = QPushButton("Run backtest")
        self.run_btn.clicked.connect(self._on_run_clicked)
        body.addWidget(self.run_btn, 5, 0)
        self.bt_status = QLabel("")
        self.bt_status.setStyleSheet("color: #bbb;")
        body.addWidget(self.bt_status, 5, 1, 1, 7)

        # PnL plot + summary
        self.pnl_plot = pg.PlotWidget()
        self.pnl_plot.setBackground("#111")
        self.pnl_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pnl_plot.setLabel("left", "Cum PnL ($)")
        self.pnl_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        bt_root.addWidget(self.pnl_plot, 2)
        self.pnl_curve = self.pnl_plot.plot(
            [], [], pen=pg.mkPen("#ffd166", width=2), name="Equity")
        self.pnl_plot.addItem(pg.InfiniteLine(pos=0, angle=0,
                                              pen=pg.mkPen("#555", style=Qt.DashLine)))

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #ddd; padding: 4px;")
        self.summary_label.setWordWrap(True)
        bt_root.addWidget(self.summary_label)

        # Start with body hidden (group collapsed)
        self._bt_body.setVisible(False)
        self.pnl_plot.setVisible(False)
        self.summary_label.setVisible(False)

        self._ticker: str | None = None

    # ------------------------------------------------------------------
    def set_ticker(self, ticker: str):
        self._ticker = ticker.upper()
        self.refresh()

    def _on_rebuild(self):
        if self._ticker:
            self.rebuild_requested.emit(self._ticker, self.dte_spin.value())

    def _on_bt_toggled(self, on: bool):
        self._bt_body.setVisible(on)
        self.pnl_plot.setVisible(on)
        self.summary_label.setVisible(on)

    def _on_structure_changed(self, text: str):
        self.bt_strangle_delta.setEnabled(text.lower() == "strangle")

    def _on_run_clicked(self):
        if not self._ticker:
            self.bt_status.setText("Load a ticker first.")
            return
        cfg = self._build_config()
        self.run_btn.setEnabled(False)
        self.bt_status.setText("Running…")
        self.backtest_requested.emit(self._ticker, cfg)

    def _build_config(self) -> StraddleBacktestConfig:
        hedge = HedgeConfig(
            use_interval=self.cb_h_int.isChecked(),
            interval_days=self.sp_h_int.value(),
            use_delta_threshold=self.cb_h_delta.isChecked(),
            delta_threshold=self.sp_h_delta.value(),
            use_spot_move=self.cb_h_spot.isChecked(),
            spot_move_pct=self.sp_h_spot.value(),
        )
        return StraddleBacktestConfig(
            structure="strangle" if self.structure_combo.currentText().lower() == "strangle" else "straddle",
            target_dte=self.bt_dte.value(),
            qty=self.bt_qty.value(),
            strangle_delta=self.bt_strangle_delta.value(),
            use_iv_rank=self.cb_ivrank.isChecked(),
            iv_rank_threshold=self.sp_ivrank_thr.value(),
            iv_rank_lookback=self.sp_ivrank_lb.value(),
            iv_rank_method="percentile" if self.cb_ivrank_method.currentText().lower() == "percentile" else "rank",
            use_vrp_filter=self.cb_vrp.isChecked(),
            vrp_threshold=self.sp_vrp_thr.value(),
            conditional_only=self.cb_conditional.isChecked(),
            use_dte_exit=self.cb_dte_exit.isChecked(),
            dte_exit_threshold=self.sp_dte_exit.value(),
            use_profit_target=self.cb_profit.isChecked(),
            profit_target_pct=self.sp_profit.value(),
            use_stop_loss=self.cb_stop.isChecked(),
            stop_loss_pct=self.sp_stop.value(),
            hedge=hedge,
        )

    def on_backtest_progress(self, cur: int, total: int, msg: str):
        self.bt_status.setText(f"{msg}  ({cur}/{total})")

    def on_backtest_result(self, result):
        self.run_btn.setEnabled(True)
        if result is None:
            self.bt_status.setText("No result.")
            return
        if result.error:
            self.bt_status.setText(f"Error: {result.error}")
            self.pnl_curve.setData([], [])
            self.summary_label.setText("")
            return
        eq = result.equity_curve
        if eq.empty:
            self.bt_status.setText("No equity curve produced.")
            self.pnl_curve.setData([], [])
            self.summary_label.setText("")
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
            f"Sharpe: {s.get('sharpe', 0):.2f}"
        )
        self.bt_status.setText(f"Done. {s.get('n_trades', 0)} trades.")

    def on_backtest_failed(self, msg: str):
        self.run_btn.setEnabled(True)
        self.bt_status.setText(f"Failed: {msg.splitlines()[0] if msg else ''}")

    # ------------------------------------------------------------------
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
