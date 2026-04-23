"""Simulation tab: straddle chain view + historical trade simulation."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import QDate, Qt, QThreadPool
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSplitter, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from ...analytics.simulation import HedgeConfig, Leg, SimResult, run_simulation
from ...data import cache
from ...pricing.rates import RateCurve
from ..workers import Worker

# Straddle table column indices
_C_IV    = 0
_C_DELTA = 1
_C_SELL  = 2   # red, clickable → sell call
_C_BUY   = 3   # green, clickable → buy call
_STRIKE  = 4
_P_BUY   = 5   # green, clickable → buy put
_P_SELL  = 6   # red, clickable → sell put
_P_DELTA = 7
_P_IV    = 8

_HEADERS = ["IV%", "Δ", "Sell", "Buy", "Strike", "Buy", "Sell", "Δ", "IV%"]

# col → (right, direction)
_CLICKABLE = {
    _C_SELL: ("C", -1),
    _C_BUY:  ("C", +1),
    _P_BUY:  ("P", +1),
    _P_SELL: ("P", -1),
}

_CLR_SELL = QColor(110, 22, 22)
_CLR_BUY  = QColor(18, 72, 28)
_CLR_ATM  = QColor(75, 62, 12)
_CLR_POS  = QColor(38, 38, 88)
_CLR_INFO = QColor(28, 28, 34)   # neutral dark for IV/Delta/Strike cells
_TXT_LIGHT = QColor(200, 200, 200)


class SimTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rates: RateCurve | None = None
        self._entry_chain: pd.DataFrame | None = None
        self._ticker: str = ""
        self._entry_date: date | None = None
        self._legs: list[Leg] = []
        self._stock_shares: int = 0
        self._strike_to_row: dict[float, int] = {}   # strike → straddle table row index

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_chain_panel())
        splitter.addWidget(self._build_side_panel())
        splitter.setSizes([400, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_rates(self, rates: RateCurve) -> None:
        self._rates = rates

    # ------------------------------------------------------------------
    # Top bar
    # ------------------------------------------------------------------

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)

        layout.addWidget(QLabel("Ticker:"))
        self.ticker_edit = QLineEdit("SPY")
        self.ticker_edit.setFixedWidth(70)
        layout.addWidget(self.ticker_edit)

        layout.addWidget(QLabel("Date:"))
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate().addDays(-30))
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setFixedWidth(110)
        layout.addWidget(self.date_edit)

        load_btn = QPushButton("Load Chain")
        load_btn.clicked.connect(self._on_load_chain)
        layout.addWidget(load_btn)

        self.chain_status = QLabel("No chain loaded.")
        layout.addWidget(self.chain_status, 1)
        return bar

    # ------------------------------------------------------------------
    # Chain panel (left)
    # ------------------------------------------------------------------

    def _build_chain_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        expiry_bar = QHBoxLayout()
        expiry_bar.addWidget(QLabel("Expiry:"))
        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(140)
        self.expiry_combo.currentIndexChanged.connect(self._on_expiry_changed)
        expiry_bar.addWidget(self.expiry_combo)
        expiry_bar.addStretch(1)
        self.spot_label = QLabel("")
        expiry_bar.addWidget(self.spot_label)
        layout.addLayout(expiry_bar)

        self.straddle_table = QTableWidget(0, 9)
        self.straddle_table.setHorizontalHeaderLabels(_HEADERS)
        self.straddle_table.verticalHeader().setVisible(False)
        self.straddle_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.straddle_table.setSelectionMode(QTableWidget.NoSelection)
        self.straddle_table.setAlternatingRowColors(False)

        hdr = self.straddle_table.horizontalHeader()
        # Info columns stretch equally; Strike fixed at center
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(_STRIKE, QHeaderView.Fixed)
        self.straddle_table.setColumnWidth(_STRIKE, 72)

        self.straddle_table.cellClicked.connect(self._on_straddle_click)
        layout.addWidget(self.straddle_table, 1)

        hint = QLabel(
            "  Click  Buy / Sell  to add legs.  "
            "Clicking the same cell again adjusts quantity."
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)
        return panel

    # ------------------------------------------------------------------
    # Side panel (right)
    # ------------------------------------------------------------------

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Controls (compact, fixed height) ──────────────────────────
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(4)

        # Position row: legs table + buttons + greeks
        pos_box = QGroupBox("Position")
        pos_layout = QVBoxLayout(pos_box)
        pos_layout.setSpacing(3)

        self.legs_table = QTableWidget(0, 4)
        self.legs_table.setHorizontalHeaderLabels(["Symbol", "Qty", "Fill $", "Expiry"])
        self.legs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.legs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.legs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.legs_table.setFixedHeight(110)
        pos_layout.addWidget(self.legs_table)

        btn_row = QHBoxLayout()
        add_stock_btn = QPushButton("+ Stock")
        add_stock_btn.setFixedHeight(24)
        add_stock_btn.clicked.connect(self._on_add_stock)
        clear_btn = QPushButton("Clear All")
        clear_btn.setFixedHeight(24)
        clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(add_stock_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(clear_btn)
        pos_layout.addLayout(btn_row)

        self.greeks_label = QLabel("Δ —  Γ —  V —  Θ —")
        self.greeks_label.setStyleSheet("font-family: monospace; color: #aaa; font-size: 11px;")
        pos_layout.addWidget(self.greeks_label)
        ctrl_layout.addWidget(pos_box)

        # Hedge + Run in one compact row
        hedge_run = QHBoxLayout()

        hedge_box = QGroupBox("Delta Hedge")
        hedge_layout = QVBoxLayout(hedge_box)
        hedge_layout.setSpacing(2)
        hedge_layout.setContentsMargins(6, 4, 6, 4)

        r1 = QHBoxLayout()
        self.cb_interval = QCheckBox("Every")
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 252)
        self.spin_interval.setValue(5)
        self.spin_interval.setFixedWidth(52)
        r1.addWidget(self.cb_interval)
        r1.addWidget(self.spin_interval)
        r1.addWidget(QLabel("days"))
        r1.addStretch()
        hedge_layout.addLayout(r1)

        r2 = QHBoxLayout()
        self.cb_delta = QCheckBox("|Δ| >")
        self.spin_delta = QDoubleSpinBox()
        self.spin_delta.setRange(1, 100000)
        self.spin_delta.setValue(100)
        self.spin_delta.setDecimals(0)
        self.spin_delta.setFixedWidth(64)
        r2.addWidget(self.cb_delta)
        r2.addWidget(self.spin_delta)
        r2.addWidget(QLabel("sh"))
        r2.addStretch()
        hedge_layout.addLayout(r2)

        r3 = QHBoxLayout()
        self.cb_spot = QCheckBox("Move >")
        self.spin_spot = QDoubleSpinBox()
        self.spin_spot.setRange(0.1, 50.0)
        self.spin_spot.setValue(2.0)
        self.spin_spot.setDecimals(1)
        self.spin_spot.setFixedWidth(56)
        r3.addWidget(self.cb_spot)
        r3.addWidget(self.spin_spot)
        r3.addWidget(QLabel("%"))
        r3.addStretch()
        hedge_layout.addLayout(r3)

        hedge_run.addWidget(hedge_box, 2)

        run_widget = QWidget()
        run_vlayout = QVBoxLayout(run_widget)
        run_vlayout.setContentsMargins(4, 0, 0, 0)
        self.run_btn = QPushButton("Run\nSimulation")
        self.run_btn.setMinimumHeight(60)
        self.run_btn.clicked.connect(self._on_run)
        run_vlayout.addWidget(self.run_btn)
        self.sim_status = QLabel("")
        self.sim_status.setWordWrap(True)
        self.sim_status.setStyleSheet("color: #aaa; font-size: 11px;")
        run_vlayout.addWidget(self.sim_status)
        run_vlayout.addStretch()
        hedge_run.addWidget(run_widget, 1)

        ctrl_layout.addLayout(hedge_run)
        layout.addWidget(controls, 0)

        # ── Charts (always visible, empty until run) ───────────────────
        charts_splitter = QSplitter(Qt.Vertical)

        self.pnl_plot = pg.PlotWidget()
        self.pnl_plot.setBackground("#111")
        self.pnl_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pnl_plot.setLabel("left", "P&L ($)")
        self.pnl_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        self._pnl_line = self.pnl_plot.plot([], [], pen=pg.mkPen("w", width=2))
        self._pnl_zero = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen("#555", style=Qt.DashLine)
        )
        self.pnl_plot.addItem(self._pnl_zero)
        charts_splitter.addWidget(self.pnl_plot)

        self.delta_plot = pg.PlotWidget()
        self.delta_plot.setBackground("#111")
        self.delta_plot.showGrid(x=True, y=True, alpha=0.3)
        self.delta_plot.setLabel("left", "Delta")
        self.delta_plot.setAxisItems({"bottom": pg.DateAxisItem()})
        self.delta_plot.setXLink(self.pnl_plot)
        self._delta_line = self.delta_plot.plot([], [], pen=pg.mkPen("#4aa3ff", width=2))
        charts_splitter.addWidget(self.delta_plot)

        charts_splitter.setSizes([200, 100])
        layout.addWidget(charts_splitter, 3)

        # ── Result tabs ────────────────────────────────────────────────
        self.result_tabs = QTabWidget()

        self.daily_table = QTableWidget(0, 5)
        self.daily_table.setHorizontalHeaderLabels(
            ["Date", "Spot", "P&L ($)", "Δ Portfolio", "Θ /day"]
        )
        self.daily_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.daily_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.daily_table.setAlternatingRowColors(True)
        self.result_tabs.addTab(self.daily_table, "Daily P&L")

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFontFamily("Courier New")
        self.result_tabs.addTab(self.summary_text, "Summary")

        self.hedge_table = QTableWidget(0, 5)
        self.hedge_table.setHorizontalHeaderLabels(
            ["Date", "Trigger", "Shares", "Spot", "Δ Before"]
        )
        self.hedge_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hedge_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_tabs.addTab(self.hedge_table, "Hedge Log")

        layout.addWidget(self.result_tabs, 2)
        return panel

    # ------------------------------------------------------------------
    # Load chain
    # ------------------------------------------------------------------

    def _on_load_chain(self) -> None:
        ticker = self.ticker_edit.text().strip().upper()
        qd = self.date_edit.date()
        day = date(qd.year(), qd.month(), qd.day())

        if not ticker:
            QMessageBox.warning(self, "Input error", "Please enter a ticker.")
            return

        if day not in set(cache.cached_chain_dates(ticker)):
            self.chain_status.setText(f"Date {day} not cached for {ticker}.")
            QMessageBox.warning(
                self, "Not cached",
                f"No cached chain for {ticker} on {day}.\n"
                "Use the Backfill button on another tab first.",
            )
            return

        chain = cache.read_chain(ticker, day)
        if chain is None or chain.empty:
            self.chain_status.setText("Chain file empty or unreadable.")
            return

        self._ticker = ticker
        self._entry_date = day
        self._entry_chain = chain
        self._legs = []
        self._stock_shares = 0

        spot = float(chain["spot"].iloc[0])
        n = len(chain)
        n_exp = chain["expiry"].nunique()
        self.chain_status.setText(
            f"{ticker}  {day}  ·  {n} contracts  ·  {n_exp} expiries  ·  spot ${spot:.2f}"
        )
        self.spot_label.setText(f"Spot: ${spot:.2f}")

        # Populate expiry selector
        expiries = sorted(chain["expiry"].dropna().unique())
        self.expiry_combo.blockSignals(True)
        self.expiry_combo.clear()
        for e in expiries:
            label = str(pd.Timestamp(e).date())
            self.expiry_combo.addItem(label, userData=e)
        self.expiry_combo.blockSignals(False)
        self.expiry_combo.setCurrentIndex(0)
        self._reload_straddle()

        # Reset position UI
        self.legs_table.setRowCount(0)
        self._update_greeks_label()

    # ------------------------------------------------------------------
    # Straddle view
    # ------------------------------------------------------------------

    def _on_expiry_changed(self) -> None:
        self._reload_straddle()

    def _reload_straddle(self) -> None:
        """Full rebuild — only called on chain load or expiry change."""
        if self._entry_chain is None:
            return
        expiry = self.expiry_combo.currentData()
        if expiry is None:
            return

        chain = self._entry_chain
        sub = chain[chain["expiry"] == expiry]
        calls = sub[sub["right"] == "C"].set_index("strike")
        puts  = sub[sub["right"] == "P"].set_index("strike")
        strikes = sorted(set(calls.index) | set(puts.index))
        spot = float(chain["spot"].iloc[0])

        self._strike_to_row = {s: i for i, s in enumerate(strikes)}
        atm_row = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot), default=0)

        pos_symbols = {leg.symbol for leg in self._legs}
        bold = QFont()
        bold.setBold(True)

        self.straddle_table.setUpdatesEnabled(False)
        self.straddle_table.setRowCount(len(strikes))

        for i, strike in enumerate(strikes):
            c = calls.loc[strike] if strike in calls.index else None
            p = puts.loc[strike] if strike in puts.index else None

            c_sym = str(c["symbol"]) if c is not None else None
            p_sym = str(p["symbol"]) if p is not None else None
            has_call = c_sym in pos_symbols if c_sym else False
            has_put  = p_sym in pos_symbols if p_sym else False

            def _fmt_iv(row):
                return f"{row['iv']*100:.1f}" if row is not None and np.isfinite(row["iv"]) else "—"
            def _fmt_delta(row):
                return f"{row['delta']:.2f}" if row is not None and np.isfinite(row["delta"]) else "—"
            def _fmt_mid(row):
                return f"{row['mid']:.2f}" if row is not None and np.isfinite(row["mid"]) else "—"

            cell_data = [
                (_fmt_iv(c),           _CLR_POS if has_call else _CLR_INFO),   # _C_IV
                (_fmt_delta(c),        _CLR_POS if has_call else _CLR_INFO),   # _C_DELTA
                (_fmt_mid(c),          _CLR_POS if has_call else _CLR_SELL),   # _C_SELL
                (_fmt_mid(c),          _CLR_POS if has_call else _CLR_BUY),    # _C_BUY
                (f"{strike:.1f}",      _CLR_ATM if i == atm_row else _CLR_INFO), # _STRIKE
                (_fmt_mid(p),          _CLR_POS if has_put  else _CLR_BUY),    # _P_BUY
                (_fmt_mid(p),          _CLR_POS if has_put  else _CLR_SELL),   # _P_SELL
                (_fmt_delta(p),        _CLR_POS if has_put  else _CLR_INFO),   # _P_DELTA
                (_fmt_iv(p),           _CLR_POS if has_put  else _CLR_INFO),   # _P_IV
            ]

            for col, (text, bg) in enumerate(cell_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(_TXT_LIGHT)
                item.setBackground(bg)
                if col == _STRIKE:
                    item.setFont(bold)
                    item.setData(Qt.UserRole, float(strike))
                self.straddle_table.setItem(i, col, item)

        self.straddle_table.setUpdatesEnabled(True)
        self.straddle_table.scrollToItem(
            self.straddle_table.item(max(atm_row - 5, 0), _STRIKE)
        )

    def _update_row_highlight(self, strike: float) -> None:
        """Fast path: update only the background of one strike row after a leg change."""
        row = self._strike_to_row.get(strike)
        if row is None:
            return
        expiry = self.expiry_combo.currentData()
        if expiry is None:
            return

        chain = self._entry_chain
        pos_symbols = {leg.symbol for leg in self._legs}

        def _sym(right):
            mask = (
                (chain["expiry"] == expiry) &
                (chain["right"] == right) &
                (chain["strike"] == strike)
            )
            rows = chain[mask]
            return str(rows.iloc[0]["symbol"]) if not rows.empty else None

        c_sym = _sym("C")
        p_sym = _sym("P")
        has_call = c_sym in pos_symbols if c_sym else False
        has_put  = p_sym in pos_symbols if p_sym else False

        # (col, bg_if_no_pos, side)  side: "C", "P", None
        col_map = [
            (_C_IV,    _CLR_INFO, "C"),
            (_C_DELTA, _CLR_INFO, "C"),
            (_C_SELL,  _CLR_SELL, "C"),
            (_C_BUY,   _CLR_BUY,  "C"),
            (_P_BUY,   _CLR_BUY,  "P"),
            (_P_SELL,  _CLR_SELL, "P"),
            (_P_DELTA, _CLR_INFO, "P"),
            (_P_IV,    _CLR_INFO, "P"),
        ]
        for col, base_clr, side in col_map:
            item = self.straddle_table.item(row, col)
            if item is None:
                continue
            has_pos = has_call if side == "C" else has_put
            item.setBackground(_CLR_POS if has_pos else base_clr)

    # ------------------------------------------------------------------
    # Straddle click → add/adjust leg
    # ------------------------------------------------------------------

    def _on_straddle_click(self, row: int, col: int) -> None:
        if col not in _CLICKABLE:
            return
        if self._entry_chain is None:
            return

        right, direction = _CLICKABLE[col]
        expiry = self.expiry_combo.currentData()

        strike_item = self.straddle_table.item(row, _STRIKE)
        if strike_item is None:
            return
        strike = float(strike_item.data(Qt.UserRole))

        mask = (
            (self._entry_chain["expiry"] == expiry) &
            (self._entry_chain["right"] == right) &
            (self._entry_chain["strike"] == strike)
        )
        rows = self._entry_chain[mask]
        if rows.empty:
            return
        row_data = rows.iloc[0]
        symbol = str(row_data["symbol"])
        price  = float(row_data["mid"])
        expiry_ts = row_data["expiry"]
        if not isinstance(expiry_ts, pd.Timestamp):
            expiry_ts = pd.Timestamp(expiry_ts)

        # Accumulate: find existing leg or add new
        for leg in self._legs:
            if leg.symbol == symbol:
                leg.qty += direction
                if leg.qty == 0:
                    self._legs.remove(leg)
                break
        else:
            self._legs.append(Leg(
                symbol=symbol,
                right=right,
                strike=strike,
                expiry=expiry_ts,
                qty=direction,
                entry_price=price,
            ))

        self._refresh_legs_table()
        self._update_row_highlight(strike)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _on_add_stock(self) -> None:
        qty, ok = QInputDialog.getInt(
            self, "Add Stock",
            "Shares (+long, −short):",
            value=100, min=-100000, max=100000,
        )
        if ok and qty != 0:
            self._stock_shares += qty
            self._refresh_legs_table()

    def _on_clear(self) -> None:
        self._legs = []
        self._stock_shares = 0
        self._refresh_legs_table()
        if self._entry_chain is not None:
            self._reload_straddle()

    def _refresh_legs_table(self) -> None:
        self.legs_table.setRowCount(0)
        for leg in self._legs:
            r = self.legs_table.rowCount()
            self.legs_table.insertRow(r)
            self.legs_table.setItem(r, 0, QTableWidgetItem(leg.symbol))
            qty_item = QTableWidgetItem(f"{leg.qty:+d}")
            qty_item.setForeground(
                QColor("#6aff9a") if leg.qty > 0 else QColor("#ff6a6a")
            )
            self.legs_table.setItem(r, 1, qty_item)
            self.legs_table.setItem(r, 2, QTableWidgetItem(f"${leg.entry_price:.2f}"))
            self.legs_table.setItem(r, 3, QTableWidgetItem(str(leg.expiry.date())))
        if self._stock_shares:
            r = self.legs_table.rowCount()
            self.legs_table.insertRow(r)
            self.legs_table.setItem(r, 0, QTableWidgetItem("STOCK"))
            qty_item = QTableWidgetItem(f"{self._stock_shares:+d} sh")
            qty_item.setForeground(
                QColor("#6aff9a") if self._stock_shares > 0 else QColor("#ff6a6a")
            )
            self.legs_table.setItem(r, 1, qty_item)
            self.legs_table.setItem(r, 2, QTableWidgetItem("—"))
            self.legs_table.setItem(r, 3, QTableWidgetItem("—"))
        self._update_greeks_label()

    def _update_greeks_label(self) -> None:
        if self._entry_chain is None or not self._legs:
            self.greeks_label.setText("Δ —  Γ —  V —  Θ —")
            return
        delta = gamma = vega = theta = 0.0
        for leg in self._legs:
            rows = self._entry_chain[self._entry_chain["symbol"] == leg.symbol]
            if rows.empty:
                continue
            row = rows.iloc[0]
            m = leg.qty * 100
            delta += m * (float(row["delta"]) if np.isfinite(row["delta"]) else 0)
            gamma += m * (float(row["gamma"]) if np.isfinite(row["gamma"]) else 0)
            vega  += m * (float(row["vega"])  if np.isfinite(row["vega"])  else 0)
            theta += m * (float(row["theta"]) if np.isfinite(row["theta"]) else 0)
        delta += self._stock_shares
        self.greeks_label.setText(
            f"Δ {delta:+.1f}  Γ {gamma:+.4f}  V {vega:+.2f}  Θ {theta:+.2f}/day"
        )

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        if not self._legs and self._stock_shares == 0:
            QMessageBox.warning(self, "No position", "Add at least one leg first.")
            return
        if self._entry_chain is None or self._entry_date is None:
            QMessageBox.warning(self, "No chain", "Load a chain first.")
            return
        if self._rates is None:
            QMessageBox.warning(self, "Not ready", "Rate curve not initialized.")
            return

        hedge_config = HedgeConfig(
            use_interval=self.cb_interval.isChecked(),
            interval_days=self.spin_interval.value(),
            use_delta_threshold=self.cb_delta.isChecked(),
            delta_threshold=self.spin_delta.value(),
            use_spot_move=self.cb_spot.isChecked(),
            spot_move_pct=self.spin_spot.value(),
        )

        self.run_btn.setEnabled(False)
        self.sim_status.setText("Running…")
        self._pnl_line.setData([], [])
        self._delta_line.setData([], [])

        worker = Worker(
            run_simulation,
            self._ticker, self._entry_date,
            list(self._legs), self._stock_shares,
            hedge_config, self._rates,
        )
        worker.signals.progress.connect(
            lambda cur, tot, msg: self.sim_status.setText(f"{msg}  ({cur}/{tot})")
        )
        worker.signals.finished.connect(self._on_result)
        worker.signals.failed.connect(self._on_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_result(self, result: SimResult) -> None:
        self.run_btn.setEnabled(True)
        if result.error:
            self.sim_status.setText(f"Error: {result.error}")
            QMessageBox.critical(self, "Simulation error", result.error)
            return
        self.sim_status.setText(f"Final P&L: ${result.final_pnl:+,.2f}")
        self._redraw(result)

    def _on_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.sim_status.setText("Simulation failed.")
        QMessageBox.critical(self, "Simulation failed", msg)

    # ------------------------------------------------------------------
    # Results rendering
    # ------------------------------------------------------------------

    def _redraw(self, result: SimResult) -> None:
        if not result.states:
            return

        xs  = np.array([pd.Timestamp(s.date).timestamp() for s in result.states])
        pnl = np.array([s.pnl for s in result.states])
        dlts = np.array([s.portfolio_delta for s in result.states])

        self._pnl_line.setData(xs, pnl)
        self._delta_line.setData(xs, dlts)

        # Hedge event markers on P&L chart
        for item in getattr(self, "_hedge_scatter_items", []):
            self.pnl_plot.removeItem(item)
        self._hedge_scatter_items = []
        if result.hedge_log:
            date_to_pnl = {s.date: s.pnl for s in result.states}
            hx = np.array([pd.Timestamp(h.date).timestamp() for h in result.hedge_log])
            hy = np.array([date_to_pnl.get(h.date, 0.0) for h in result.hedge_log])
            sc = pg.ScatterPlotItem(
                x=hx, y=hy, symbol="t", size=10,
                pen=pg.mkPen("#ffaa00"), brush=pg.mkBrush("#ffaa00"),
            )
            self.pnl_plot.addItem(sc)
            self._hedge_scatter_items.append(sc)


        # Summary
        eg = result.entry_greeks
        legs_lines = "\n".join(
            f"  {'BUY ' if l.qty > 0 else 'SELL'} {abs(l.qty):2d}x  "
            f"{l.symbol}  @ ${l.entry_price:.2f}"
            for l in result.legs
        )
        hedge_info = (
            f"{len(result.hedge_log)} trade(s), last {result.hedge_log[-1].date}"
            if result.hedge_log else "none"
        )
        self.summary_text.setPlainText(
            f"Ticker:        {result.ticker}\n"
            f"Entry date:    {result.entry_date}\n"
            f"Trading days:  {len(result.states)}\n"
            f"\nLegs:\n{legs_lines}\n"
            f"\nEntry greeks (portfolio, ×100 per contract):\n"
            f"  Delta  {eg.get('delta', 0):+.2f}\n"
            f"  Gamma  {eg.get('gamma', 0):+.4f}\n"
            f"  Vega   {eg.get('vega',  0):+.2f}\n"
            f"  Theta  {eg.get('theta', 0):+.2f} /day\n"
            f"\nHedge trades:  {hedge_info}\n"
            f"\nFinal P&L:     ${result.final_pnl:+,.2f}"
        )

        # Daily P&L table
        green = QColor("#4aff7a")
        red   = QColor("#ff5555")
        self.daily_table.setUpdatesEnabled(False)
        self.daily_table.setRowCount(len(result.states))
        for r, s in enumerate(result.states):
            pnl_color = green if s.pnl >= 0 else red
            cells = [
                str(s.date),
                f"${s.spot:.2f}",
                f"${s.pnl:+,.2f}",
                f"{s.portfolio_delta:+.1f}",
                f"{s.portfolio_theta:+.2f}",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 2:
                    item.setForeground(pnl_color)
                self.daily_table.setItem(r, c, item)
        self.daily_table.setUpdatesEnabled(True)
        self.result_tabs.setCurrentIndex(0)   # jump to Daily P&L tab

        # Hedge log table
        self.hedge_table.setRowCount(0)
        for h in result.hedge_log:
            r = self.hedge_table.rowCount()
            self.hedge_table.insertRow(r)
            self.hedge_table.setItem(r, 0, QTableWidgetItem(str(h.date)))
            self.hedge_table.setItem(r, 1, QTableWidgetItem(h.trigger))
            self.hedge_table.setItem(r, 2, QTableWidgetItem(f"{h.shares_traded:+d}"))
            self.hedge_table.setItem(r, 3, QTableWidgetItem(f"${h.spot:.2f}"))
            self.hedge_table.setItem(r, 4, QTableWidgetItem(f"{h.portfolio_delta_before:+.1f}"))
