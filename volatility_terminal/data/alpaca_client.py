"""Thin wrapper around alpaca-py for historical options + equity data.

Free-tier friendly: rate-limited, retry on 429, paginates contract lists.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd

_OCC_RE = re.compile(r"^[A-Z]{1,5}\d{6,7}[CP]\d{8}$")

try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import AssetStatus, ContractType
except ImportError as e:
    raise ImportError(
        "alpaca-py is required. Install with `pip install alpaca-py`."
    ) from e

try:
    from alpaca.data.enums import OptionsFeed
    _HAS_OPTIONS_FEED = True
except ImportError:
    OptionsFeed = None
    _HAS_OPTIONS_FEED = False

_RECENT_WINDOW = timedelta(minutes=20)


@dataclass
class AlpacaCreds:
    api_key: str
    api_secret: str


class AlpacaOptionsData:
    """Fetches option contracts and EOD bars for a ticker from Alpaca.

    Paper trading client is used for the contracts endpoint since free-tier
    market-data keys have read access via the trading API's contracts endpoint.
    """

    SYMBOL_BATCH = 100          # option bars request: symbols per call
    RATE_LIMIT_SLEEP = 0.40     # ~150 req/min, under the 200/min free-tier cap

    def __init__(self, creds: AlpacaCreds):
        self.creds = creds
        self._opt = OptionHistoricalDataClient(creds.api_key, creds.api_secret)
        self._stk = StockHistoricalDataClient(creds.api_key, creds.api_secret)
        self._trd = TradingClient(creds.api_key, creds.api_secret, paper=True)

    # -- contracts --------------------------------------------------------

    def list_contracts(self, underlying: str, as_of: date,
                       expiration_after: date | None = None,
                       expiration_before: date | None = None) -> pd.DataFrame:
        """List option contracts for ``underlying`` that were listed as of ``as_of``.

        Filters to contracts whose expiration_date is >= as_of by default so we
        only consider chains that were live on ``as_of``.
        """
        exp_after = expiration_after or as_of
        exp_before = expiration_before
        all_rows = []
        # Query both ACTIVE and INACTIVE: a historical chain contains
        # contracts that have since expired (now INACTIVE) alongside those
        # still alive today (ACTIVE).
        for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
            page_token = None
            while True:
                req = GetOptionContractsRequest(
                    underlying_symbols=[underlying],
                    status=status,
                    expiration_date_gte=exp_after,
                    expiration_date_lte=exp_before,
                    limit=10_000,
                    page_token=page_token,
                )
                resp = self._trd.get_option_contracts(req)
                contracts = resp.option_contracts or []
                for c in contracts:
                    all_rows.append({
                        "symbol": c.symbol,
                        "underlying": c.underlying_symbol,
                        "expiration": pd.Timestamp(c.expiration_date),
                        "strike": float(c.strike_price),
                        "right": "C" if c.type == ContractType.CALL else "P",
                        "style": getattr(c.style, "value", str(c.style)),
                    })
                page_token = getattr(resp, "next_page_token", None)
                if not page_token:
                    break
                time.sleep(self.RATE_LIMIT_SLEEP)
        if not all_rows:
            return pd.DataFrame(columns=[
                "symbol", "underlying", "expiration", "strike", "right", "style"
            ])
        return pd.DataFrame(all_rows).drop_duplicates("symbol").reset_index(drop=True)

    # -- option bars ------------------------------------------------------

    def get_daily_bars(self, symbols: list[str], day: date) -> pd.DataFrame:
        """Daily OHLC bars for each option symbol on ``day``.

        Returns DataFrame with columns [symbol, open, high, low, close, volume,
        trade_count, vwap]. Missing/untraded contracts are absent.
        """
        symbols = [s for s in symbols if _OCC_RE.match(s)]
        if not symbols:
            return pd.DataFrame()
        start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        # Free-tier OPRA requires end to be ~15+ min in the past.
        cutoff = datetime.now(timezone.utc) - _RECENT_WINDOW
        if end > cutoff:
            end = cutoff
        if end <= start:
            return pd.DataFrame()
        frames = []
        for i in range(0, len(symbols), self.SYMBOL_BATCH):
            batch = symbols[i:i + self.SYMBOL_BATCH]
            req_kwargs = dict(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            if _HAS_OPTIONS_FEED:
                req_kwargs["feed"] = OptionsFeed.OPRA
            req = OptionBarsRequest(**req_kwargs)
            df = self._request_with_retry(
                lambda: self._opt.get_option_bars(req).df
            )
            if df is not None and len(df):
                frames.append(df)
            time.sleep(self.RATE_LIMIT_SLEEP)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames).reset_index()
        keep = [c for c in ["symbol", "timestamp", "open", "high", "low",
                            "close", "volume", "trade_count", "vwap"]
                if c in df.columns]
        return df[keep]

    def get_bars_range(self, symbols: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Daily bars for each option symbol across a date range.

        Returns DataFrame with columns [symbol, timestamp, open, high, low,
        close, volume, trade_count, vwap] covering all trading days in the range.
        Prefer this over repeated get_daily_bars calls during backfill.
        """
        symbols = [s for s in symbols if _OCC_RE.match(s)]
        if not symbols:
            return pd.DataFrame()
        start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - _RECENT_WINDOW
        if end > cutoff:
            end = cutoff
        if end <= start:
            return pd.DataFrame()
        frames = []
        for i in range(0, len(symbols), self.SYMBOL_BATCH):
            batch = symbols[i:i + self.SYMBOL_BATCH]
            req_kwargs = dict(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            if _HAS_OPTIONS_FEED:
                req_kwargs["feed"] = OptionsFeed.OPRA
            req = OptionBarsRequest(**req_kwargs)
            df = self._request_with_retry(
                lambda: self._opt.get_option_bars(req).df
            )
            if df is not None and len(df):
                frames.append(df)
            time.sleep(self.RATE_LIMIT_SLEEP)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames).reset_index()
        keep = [c for c in ["symbol", "timestamp", "open", "high", "low",
                            "close", "volume", "trade_count", "vwap"]
                if c in df.columns]
        return df[keep]

    # -- stock bars -------------------------------------------------------

    def get_daily_stock_bars(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Daily OHLC for the underlying equity between ``start`` and ``end`` (inclusive).

        Uses the IEX feed — free-tier keys cannot query SIP.
        """
        end_ts = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - _RECENT_WINDOW
        if end_ts > cutoff:
            end_ts = cutoff
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            end=end_ts,
            feed=DataFeed.IEX,
        )
        df = self._request_with_retry(lambda: self._stk.get_stock_bars(req).df)
        if df is None or not len(df):
            return pd.DataFrame()
        return df.reset_index()

    # -- internals --------------------------------------------------------

    def _request_with_retry(self, fn, *, max_tries: int = 5):
        delay = 1.0
        for attempt in range(max_tries):
            try:
                return fn()
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "too many" in msg:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if attempt == max_tries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return None
