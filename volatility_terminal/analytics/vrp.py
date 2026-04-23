"""Variance risk premium = IV - realized vol at matched horizon."""
from __future__ import annotations

import pandas as pd


def compute_vrp(iv_ts: pd.DataFrame, rv: pd.Series) -> pd.DataFrame:
    """Join ATM-IV timeseries with realized vol, compute VRP = IV - RV.

    iv_ts : DataFrame with columns [date (UTC), atm_iv, spot]
    rv : Series indexed by date (tz-naive) in decimal annualized units.
    """
    out = iv_ts.copy()
    if out.empty:
        out["rv"] = []
        out["vrp"] = []
        return out
    out["date_key"] = pd.to_datetime(out["date"]).dt.tz_convert(None).dt.normalize()
    rv_df = rv.rename("rv").to_frame()
    rv_df.index = pd.to_datetime(rv_df.index)
    rv_df.index = rv_df.index.tz_localize(None) if rv_df.index.tz is not None else rv_df.index
    rv_df.index = rv_df.index.normalize()
    out = out.merge(rv_df, left_on="date_key", right_index=True, how="left")
    out["vrp"] = out["atm_iv"] - out["rv"]
    return out.drop(columns=["date_key"])
