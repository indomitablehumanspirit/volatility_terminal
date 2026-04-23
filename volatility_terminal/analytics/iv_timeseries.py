"""Build a rolling ATM-IV-at-fixed-DTE timeseries across cached chain dates."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..data import cache
from .term import atm_iv_at_dte


def build_iv_timeseries(ticker: str, target_dte: int,
                        dates: list[date] | None = None,
                        force_rebuild: bool = False) -> pd.DataFrame:
    """ATM IV at ``target_dte`` days across every cached chain for ``ticker``.

    Parquet-cached; returns DataFrame [date (UTC), atm_iv, spot].
    """
    ticker = ticker.upper()
    path = cache.derived_iv_ts_path(ticker, target_dte)
    if path.exists() and not force_rebuild:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    use_dates = dates if dates is not None else cache.cached_chain_dates(ticker)
    rows = []
    for d in use_dates:
        chain = cache.read_chain(ticker, d)
        if chain is None or chain.empty:
            rows.append({"date": pd.Timestamp(d, tz="UTC"),
                         "atm_iv": np.nan, "spot": np.nan})
            continue
        iv = atm_iv_at_dte(chain, target_dte)
        spot = float(chain["spot"].iloc[0]) if "spot" in chain.columns else np.nan
        rows.append({"date": pd.Timestamp(d, tz="UTC"),
                     "atm_iv": iv, "spot": spot})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
