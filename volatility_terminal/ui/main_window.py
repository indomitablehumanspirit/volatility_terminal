"""Main application window: ticker bar + tabbed analytics."""
from __future__ import annotations

from datetime import date

import pandas as pd
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import (
    QAction, QMainWindow, QMessageBox, QStatusBar, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..data.alpaca_client import AlpacaCreds, AlpacaOptionsData
from ..data import cache
from ..data.chain_fetcher import ChainFetcher
from ..pricing.rates import RateCurve
from .creds_dialog import CredsDialog, ensure_creds, save_creds
from .tabs.skew_tab import SkewTab
from .tabs.term_tab import TermTab
from .tabs.vrp_tab import VrpTab
from .tabs.sim_tab import SimTab
from .tabs.earnings_tab import EarningsTab
from .tabs.backtest_tab import BacktestTab
from .ticker_bar import TickerBar
from .workers import Worker


class MainWindow(QMainWindow):
    def __init__(self, creds: AlpacaCreds):
        super().__init__()
        self.setWindowTitle("Volatility Terminal")
        self.resize(1400, 860)

        self.alpaca = AlpacaOptionsData(creds)
        self.rates = RateCurve(cache_path=cache.rates_path())
        self.fetcher = ChainFetcher(self.alpaca, self.rates)
        self.pool = QThreadPool.globalInstance()

        self.ticker_bar = TickerBar()
        self.ticker_bar.load_requested.connect(self._on_load)
        self.ticker_bar.backfill_requested.connect(self._on_backfill)

        self.tabs = QTabWidget()
        self.term_tab = TermTab()
        self.skew_tab = SkewTab()
        self.vrp_tab = VrpTab()
        self.tabs.addTab(self.term_tab, "Term")
        self.tabs.addTab(self.skew_tab, "Skew")
        self.tabs.addTab(self.vrp_tab, "VRP")
        self.vrp_tab.rebuild_requested.connect(self._on_vrp_rebuild)
        self.vrp_tab.backtest_requested.connect(self._on_vrp_backtest)
        self.sim_tab = SimTab()
        self.sim_tab.set_rates(self.rates)
        self.tabs.addTab(self.sim_tab, "Sim")
        self.earnings_tab = EarningsTab(self.fetcher)
        self.tabs.addTab(self.earnings_tab, "Earnings")
        self.backtest_tab = BacktestTab()
        self.backtest_tab.backtest_requested.connect(self._on_backtest_run)
        self.backtest_tab.tune_requested.connect(self._on_tune_run)
        self.tabs.addTab(self.backtest_tab, "Backtest")

        # Wire comparison panels for Term and Skew tabs
        self.term_tab.comparison_panel.load_requested.connect(
            lambda eid, t, d: self._on_comparison_load(self.term_tab, eid, t, d)
        )
        self.term_tab.comparison_panel.entry_removed.connect(
            self.term_tab.remove_comparison
        )
        self.skew_tab.comparison_panel.load_requested.connect(
            lambda eid, t, d: self._on_comparison_load(self.skew_tab, eid, t, d)
        )
        self.skew_tab.comparison_panel.entry_removed.connect(
            self.skew_tab.remove_comparison
        )

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ticker_bar)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Enter a ticker and press Load.")

        self._build_menu()

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")
        act_creds = QAction("Change Alpaca credentials…", self)
        act_creds.triggered.connect(self._on_change_creds)
        menu.addAction(act_creds)
        menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        menu.addAction(act_quit)

    def _on_change_creds(self):
        existing = (self.alpaca.creds.api_key, self.alpaca.creds.api_secret)
        dlg = CredsDialog(self, existing=existing)
        if dlg.exec_():
            k, s = dlg.values()
            if k and s:
                save_creds(k, s)
                self.alpaca = AlpacaOptionsData(AlpacaCreds(k, s))
                self.fetcher = ChainFetcher(self.alpaca, self.rates)
                self.statusBar().showMessage("Credentials updated.")

    def _on_load(self, ticker: str, day: date):
        self.statusBar().showMessage(f"Fetching {ticker} chain for {day}…")
        worker = Worker(self.fetcher.get_chain, ticker, day)
        worker.signals.finished.connect(
            lambda df: self._on_chain_ready(ticker, day, df)
        )
        worker.signals.failed.connect(self._on_fetch_failed)
        self.pool.start(worker)

    def _on_chain_ready(self, ticker: str, day: date, chain: pd.DataFrame):
        if chain is None or chain.empty:
            self.statusBar().showMessage(
                f"{ticker} {day}: no chain data returned."
            )
            return
        self.term_tab.set_chain(ticker, day, chain)
        self.skew_tab.set_chain(ticker, day, chain)
        self.vrp_tab.set_ticker(ticker)
        self.earnings_tab.set_ticker(ticker)
        self.backtest_tab.set_ticker(ticker)
        self.statusBar().showMessage(
            f"{ticker} {day}: {len(chain)} contracts, "
            f"{chain['expiry'].nunique()} expiries, spot=${chain['spot'].iloc[0]:.2f}"
        )

    def _on_fetch_failed(self, msg: str):
        self.statusBar().showMessage("Fetch failed (see dialog).")
        QMessageBox.critical(self, "Fetch failed", msg)

    def _on_comparison_load(self, tab, entry_id: int, ticker: str, day: date):
        self.statusBar().showMessage(f"Fetching comparison {ticker} {day}…")
        worker = Worker(self.fetcher.get_chain, ticker, day)
        worker.signals.finished.connect(
            lambda chain: self._on_comparison_ready(tab, entry_id, ticker, day, chain)
        )
        worker.signals.failed.connect(
            lambda msg: (
                tab.comparison_panel.set_entry_status(entry_id, "failed"),
                self.statusBar().showMessage(
                    f"Comparison fetch failed: {ticker} {day}"
                ),
            )
        )
        self.pool.start(worker)

    def _on_comparison_ready(
        self, tab, entry_id: int, ticker: str, day: date, chain: pd.DataFrame
    ):
        if chain is None or chain.empty:
            tab.comparison_panel.set_entry_status(entry_id, "failed")
            self.statusBar().showMessage(f"No data for {ticker} {day}.")
            return
        tab.add_comparison(entry_id, ticker, day, chain)
        tab.comparison_panel.set_entry_status(entry_id, "ready")
        self.statusBar().showMessage(f"Comparison loaded: {ticker} {day}.")

    def _on_backfill(self, ticker: str, start: date, end: date):
        self.statusBar().showMessage(f"Backfilling {ticker} {start} → {end}…")

        def run(progress_cb=None):
            return self.fetcher.backfill_range(ticker, start, end,
                                               progress_cb=progress_cb)

        worker = Worker(run)
        worker.signals.progress.connect(
            lambda cur, tot, msg: self.statusBar().showMessage(
                f"Backfill {msg} — {cur}/{tot}"
            )
        )
        worker.signals.finished.connect(
            lambda n: self.statusBar().showMessage(f"Backfill complete ({n} days).")
        )
        worker.signals.failed.connect(self._on_fetch_failed)
        self.pool.start(worker)

    def _on_vrp_backtest(self, ticker: str, cfg):
        from ..analytics.straddle_backtest import run_straddle_backtest
        self.statusBar().showMessage(f"Running short-vol backtest for {ticker}…")

        def run(progress_cb=None):
            return run_straddle_backtest(ticker, cfg, self.rates,
                                         progress_cb=progress_cb)

        worker = Worker(run)
        worker.signals.progress.connect(self.vrp_tab.on_backtest_progress)
        worker.signals.finished.connect(self.vrp_tab.on_backtest_result)
        worker.signals.finished.connect(lambda _r: self.statusBar().showMessage(
            f"Short-vol backtest complete for {ticker}."
        ))
        worker.signals.failed.connect(self.vrp_tab.on_backtest_failed)
        worker.signals.failed.connect(lambda _m: self.statusBar().showMessage(
            "Backtest failed (see tab)."
        ))
        self.pool.start(worker)

    def _on_backtest_run(self, ticker: str, cfg):
        from ..analytics.backtest_engine import run_backtest
        self.statusBar().showMessage(f"Running backtest for {ticker}…")

        def run(progress_cb=None):
            return run_backtest(ticker, cfg, self.rates, progress_cb=progress_cb)

        worker = Worker(run)
        worker.signals.progress.connect(self.backtest_tab.on_backtest_progress)
        worker.signals.finished.connect(self.backtest_tab.on_backtest_result)
        worker.signals.finished.connect(lambda _r: self.statusBar().showMessage(
            f"Backtest complete for {ticker}."
        ))
        worker.signals.failed.connect(self.backtest_tab.on_backtest_failed)
        worker.signals.failed.connect(lambda _m: self.statusBar().showMessage(
            "Backtest failed (see tab)."
        ))
        self.pool.start(worker)

    def _on_tune_run(self, ticker: str, base_cfg, tuning_cfg):
        from ..analytics.tuning import run_tuning
        self.statusBar().showMessage(f"Tuning backtest for {ticker}…")

        def run(progress_cb=None):
            return run_tuning(ticker, base_cfg, tuning_cfg, self.rates,
                              progress_cb=progress_cb)

        worker = Worker(run)
        worker.signals.progress.connect(self.backtest_tab.on_tune_progress)
        worker.signals.finished.connect(self.backtest_tab.on_tune_result)
        worker.signals.finished.connect(lambda _r: self.statusBar().showMessage(
            f"Tuning complete for {ticker}."
        ))
        worker.signals.failed.connect(self.backtest_tab.on_tune_failed)
        worker.signals.failed.connect(lambda _m: self.statusBar().showMessage(
            "Tuning failed (see tab)."
        ))
        self.pool.start(worker)

    def _on_vrp_rebuild(self, ticker: str, dte: int):
        from ..analytics.iv_timeseries import build_iv_timeseries
        self.statusBar().showMessage(f"Rebuilding IV TS for {ticker} DTE={dte}…")

        def run():
            return build_iv_timeseries(ticker, dte, force_rebuild=True)

        worker = Worker(run)
        worker.signals.finished.connect(lambda _df: (
            self.vrp_tab.refresh(),
            self.statusBar().showMessage(f"IV TS rebuilt for {ticker} DTE={dte}."),
        ))
        worker.signals.failed.connect(self._on_fetch_failed)
        self.pool.start(worker)
