"""Main application window: ticker bar + tabbed analytics."""
from __future__ import annotations

from datetime import date

import pandas as pd
from PyQt5.QtCore import QThreadPool, Qt
from PyQt5.QtWidgets import (
    QAction, QLabel, QMainWindow, QMessageBox, QStatusBar, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..data.alpaca_client import AlpacaCreds, AlpacaOptionsData
from ..data import cache
from ..data.chain_fetcher import ChainFetcher
from ..pricing.rates import RateCurve
from .creds_dialog import CredsDialog, ensure_creds, save_creds
from .tabs.skew_tab import SkewTab
from .tabs.surface_tab import SurfaceTab
from .tabs.term_tab import TermTab
from .tabs.vrp_tab import VrpTab
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
        self.surface_tab = SurfaceTab()
        self.vrp_tab = VrpTab()
        self.tabs.addTab(self.term_tab, "Term")
        self.tabs.addTab(self.skew_tab, "Skew")
        self.tabs.addTab(self.surface_tab, "Surface")
        self.tabs.addTab(self.vrp_tab, "VRP")
        self.vrp_tab.rebuild_requested.connect(self._on_vrp_rebuild)

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
        worker.signals.finished.connect(lambda df: self._on_chain_ready(ticker, day, df))
        worker.signals.failed.connect(self._on_fetch_failed)
        self.pool.start(worker)

    def _on_chain_ready(self, ticker: str, day: date, chain: pd.DataFrame):
        if chain is None or chain.empty:
            self.statusBar().showMessage(
                f"{ticker} {day}: no chain data returned."
            )
            return
        self.term_tab.set_chain(chain)
        self.skew_tab.set_chain(chain)
        self.surface_tab.set_chain(chain)
        self.vrp_tab.set_ticker(ticker)
        self.statusBar().showMessage(
            f"{ticker} {day}: {len(chain)} contracts, "
            f"{chain['expiry'].nunique()} expiries, spot=${chain['spot'].iloc[0]:.2f}"
        )

    def _on_fetch_failed(self, msg: str):
        self.statusBar().showMessage("Fetch failed (see dialog).")
        QMessageBox.critical(self, "Fetch failed", msg)

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
