"""Historical earnings dates via yfinance, cached to Parquet."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from . import cache


def _classify_time(ts: pd.Timestamp) -> str:
    """BMO (before open), AMC (after close), or UNKNOWN.

    Yahoo's earnings timestamp is localized to US/Eastern. <= 09:30 ET → BMO,
    >= 16:00 ET → AMC; anything else (rare midday prints) → UNKNOWN and we
    conservatively treat it like AMC (close-of-day → next close).
    """
    if ts is None or pd.isna(ts):
        return "UNKNOWN"
    try:
        et = ts.tz_convert("America/New_York")
    except (TypeError, AttributeError):
        return "UNKNOWN"
    hm = et.hour * 60 + et.minute
    if hm <= 9 * 60 + 30:
        return "BMO"
    if hm >= 16 * 60:
        return "AMC"
    return "UNKNOWN"


def fetch_earnings_dates(ticker: str, limit: int = 40) -> pd.DataFrame:
    """Fetch earnings dates (past + scheduled) from yfinance.

    Returns DataFrame with columns [date, time_of_day, ts] where `date` is a
    python ``date``, `time_of_day` is 'BMO' | 'AMC' | 'UNKNOWN', and `ts` is
    the original tz-aware pandas Timestamp.
    """
    import yfinance as yf  # lazy import; optional dependency
    tk = yf.Ticker(ticker.upper())
    df = tk.get_earnings_dates(limit=limit)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "time_of_day", "ts"])
    ts = df.index.to_series().reset_index(drop=True)
    out = pd.DataFrame({"ts": ts})
    out["date"] = out["ts"].dt.tz_convert("America/New_York").dt.date
    out["time_of_day"] = out["ts"].apply(_classify_time)
    return out[["date", "time_of_day", "ts"]].sort_values("date").reset_index(drop=True)


def get_earnings_dates(ticker: str,
                       use_cache: bool = True,
                       limit: int = 40) -> pd.DataFrame:
    """Return earnings dates, preferring the local Parquet cache."""
    ticker = ticker.upper()
    if use_cache:
        cached = cache.read_earnings(ticker)
        if cached is not None and not cached.empty:
            return cached
    fetched = fetch_earnings_dates(ticker, limit=limit)
    if not fetched.empty:
        cache.write_earnings(ticker, fetched)
    return fetched


def refresh_earnings_dates(ticker: str, limit: int = 40) -> pd.DataFrame:
    """Force a re-fetch and overwrite cache."""
    return get_earnings_dates(ticker, use_cache=False, limit=limit)
