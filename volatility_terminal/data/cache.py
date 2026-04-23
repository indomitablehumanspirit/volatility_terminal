"""Parquet cache: one chain per (ticker, date); daily underlying OHLC per ticker."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache"


def chain_path(ticker: str, day: date) -> Path:
    return CACHE_ROOT / "chains" / ticker.upper() / f"{day.isoformat()}.parquet"


def underlying_path(ticker: str) -> Path:
    return CACHE_ROOT / "underlying" / f"{ticker.upper()}.parquet"


def rates_path() -> Path:
    return CACHE_ROOT / "rates" / "fred_curve.parquet"


def derived_iv_ts_path(ticker: str, dte: int) -> Path:
    return CACHE_ROOT / "derived" / f"iv_ts_{ticker.upper()}_DTE{int(dte)}.parquet"


def read_chain(ticker: str, day: date) -> pd.DataFrame | None:
    p = chain_path(ticker, day)
    return pd.read_parquet(p) if p.exists() else None


def write_chain(ticker: str, day: date, df: pd.DataFrame) -> None:
    p = chain_path(ticker, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def cached_chain_dates(ticker: str) -> list[date]:
    d = CACHE_ROOT / "chains" / ticker.upper()
    if not d.exists():
        return []
    out = []
    for f in d.glob("*.parquet"):
        try:
            out.append(date.fromisoformat(f.stem))
        except ValueError:
            continue
    return sorted(out)


def read_underlying(ticker: str) -> pd.DataFrame | None:
    p = underlying_path(ticker)
    return pd.read_parquet(p) if p.exists() else None


def write_underlying(ticker: str, df: pd.DataFrame) -> None:
    p = underlying_path(ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
