"""FRED daily Treasury yield curve: fetch, cache, interpolate r(T).

Uses FRED's public fredgraph CSV endpoint (no API key required).
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED_SERIES = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 3 / 12,
    "DGS6MO": 6 / 12,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS3":   3.0,
    "DGS5":   5.0,
    "DGS7":   7.0,
    "DGS10":  10.0,
}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(FRED_SERIES)


class RateCurve:
    """Daily Treasury par-yield curve indexed by date and tenor (years).

    Values are decimal (e.g. 0.0453), forward-filled across non-business days
    and missing observations.
    """

    def __init__(self, cache_path: Path, max_age_days: int = 1):
        self.cache_path = Path(cache_path)
        self.max_age_days = max_age_days
        self._df: pd.DataFrame | None = None

    def _load_or_fetch(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        need_fetch = True
        if self.cache_path.exists():
            age_days = (pd.Timestamp.utcnow().normalize()
                        - pd.Timestamp(self.cache_path.stat().st_mtime, unit="s", tz="UTC").normalize()
                        ).days
            if age_days <= self.max_age_days:
                need_fetch = False
        if need_fetch:
            try:
                self._df = self._fetch()
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._df.to_parquet(self.cache_path)
            except Exception:
                if self.cache_path.exists():
                    self._df = pd.read_parquet(self.cache_path)
                else:
                    raise
        else:
            self._df = pd.read_parquet(self.cache_path)
        return self._df

    @staticmethod
    def _fetch() -> pd.DataFrame:
        resp = requests.get(FRED_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
        df = df.ffill()
        return df

    def r_at(self, date, tenor_years: float) -> float:
        """Linearly interpolate the yield curve at ``tenor_years`` for ``date``.

        Date is forward-filled to the last available observation.
        """
        df = self._load_or_fetch()
        date = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo else pd.Timestamp(date)
        if date > df.index.max():
            row = df.iloc[-1]
        else:
            idx = df.index.searchsorted(date, side="right") - 1
            if idx < 0:
                return 0.045
            row = df.iloc[idx]
        tenors = np.array([FRED_SERIES[c] for c in df.columns])
        values = row.values.astype(float)
        mask = np.isfinite(values)
        if not mask.any():
            return 0.045
        tenors = tenors[mask]
        values = values[mask]
        order = np.argsort(tenors)
        tenors = tenors[order]
        values = values[order]
        t = float(np.clip(tenor_years, tenors.min(), tenors.max()))
        return float(np.interp(t, tenors, values))

    def curve_for_date(self, date) -> pd.Series:
        """Return the full curve as a Series (index=tenor_years, values=rate)."""
        df = self._load_or_fetch()
        date = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo else pd.Timestamp(date)
        idx = df.index.searchsorted(date, side="right") - 1
        idx = max(idx, 0)
        row = df.iloc[idx]
        out = pd.Series(
            row.values.astype(float),
            index=[FRED_SERIES[c] for c in df.columns],
            name=df.index[idx],
        )
        return out.sort_index().dropna()
