"""Concrete predefined signals.

Each wraps existing analytics. Per-day timeseries that aren't already on disk
are built once by walking ``cache.cached_chain_dates(ticker)``; results are
memoized in :mod:`.cache` keyed on ``signal.hash_key()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...data import cache as _cache
from ..forward_vol import forward_vol
from ..iv_timeseries import build_iv_timeseries
from ..realized import close_to_close
from ..skew_metrics import _interp_iv_at_delta, delta_skew_metrics
from ..term import term_structure
from .base import Signal


def _norm_index(s: pd.Series) -> pd.Series:
    """Normalize a Series to a tz-naive midnight-normalized DatetimeIndex."""
    if s.empty:
        return s
    idx = pd.to_datetime(s.index)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_convert(None)
    s = s.copy()
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _underlying_close_series(ticker: str) -> pd.Series:
    under = _cache.read_underlying(ticker)
    if under is None or under.empty:
        return pd.Series(dtype=float)
    df = under.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df["close"].astype(float)


# ---------------------------------------------------------------------------

class AtmIvSignal(Signal):
    KIND = "atm_iv"

    def __init__(self, dte: int = 30):
        self.dte = int(dte)

    def _compute(self, ticker: str) -> pd.Series:
        try:
            df = build_iv_timeseries(ticker, self.dte)
        except Exception:
            return pd.Series(dtype=float)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = pd.Series(df["atm_iv"].values, index=pd.to_datetime(df["date"]), name=self.label())
        return _norm_index(s).astype(float)

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "dte": self.dte}

    @classmethod
    def from_dict(cls, d: dict) -> "AtmIvSignal":
        return cls(dte=int(d.get("dte", 30)))

    def label(self) -> str:
        return f"ATM_IV_{self.dte}"


class RealizedVolSignal(Signal):
    KIND = "realized_vol"

    def __init__(self, window: int = 30):
        self.window = int(window)

    def _compute(self, ticker: str) -> pd.Series:
        close = _underlying_close_series(ticker)
        if close.empty:
            return pd.Series(dtype=float)
        rv = close_to_close(close, window=self.window)
        rv.name = self.label()
        return _norm_index(rv).astype(float)

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "window": self.window}

    @classmethod
    def from_dict(cls, d: dict) -> "RealizedVolSignal":
        return cls(window=int(d.get("window", 30)))

    def label(self) -> str:
        return f"RV_{self.window}"


class IvRvRatioSignal(Signal):
    KIND = "iv_rv_ratio"

    def __init__(self, dte: int = 30, window: int = 30):
        self.dte = int(dte)
        self.window = int(window)

    def _compute(self, ticker: str) -> pd.Series:
        iv = AtmIvSignal(self.dte).series(ticker)
        rv = RealizedVolSignal(self.window).series(ticker)
        if iv.empty or rv.empty:
            return pd.Series(dtype=float)
        idx = iv.index.union(rv.index)
        a = iv.reindex(idx)
        b = rv.reindex(idx).replace(0, np.nan)
        out = (a / b).rename(self.label())
        return out

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "dte": self.dte, "window": self.window}

    @classmethod
    def from_dict(cls, d: dict) -> "IvRvRatioSignal":
        return cls(dte=int(d.get("dte", 30)), window=int(d.get("window", 30)))

    def label(self) -> str:
        return f"IV{self.dte}_over_RV{self.window}"


class VrpSignal(Signal):
    KIND = "vrp"

    def __init__(self, dte: int = 30, window: int = 30):
        self.dte = int(dte)
        self.window = int(window)

    def _compute(self, ticker: str) -> pd.Series:
        iv = AtmIvSignal(self.dte).series(ticker)
        rv = RealizedVolSignal(self.window).series(ticker)
        if iv.empty or rv.empty:
            return pd.Series(dtype=float)
        idx = iv.index.union(rv.index)
        a = iv.reindex(idx)
        b = rv.reindex(idx)
        return (a - b).rename(self.label())

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "dte": self.dte, "window": self.window}

    @classmethod
    def from_dict(cls, d: dict) -> "VrpSignal":
        return cls(dte=int(d.get("dte", 30)), window=int(d.get("window", 30)))

    def label(self) -> str:
        return f"VRP_{self.dte}_{self.window}"


# --- skew + forward vol need per-day chain walk ---------------------------

def _skew_daily(ticker: str) -> pd.DataFrame:
    """Walk cached chains and compute per-expiry skew metrics for each date.

    Cached at module level (in addition to the signal-level cache) so that
    multiple ``RiskReversalSignal``/``ButterflySignal`` instances with different
    parameters share the underlying computation.
    """
    return _walk_chains(ticker, "_skew_daily", _skew_one)


def _term_daily(ticker: str) -> pd.DataFrame:
    return _walk_chains(ticker, "_term_daily", _term_one)


def _skew_one(chain: pd.DataFrame) -> pd.DataFrame:
    return delta_skew_metrics(chain)


def _per_expiry_skew_at_delta(chain: pd.DataFrame, target_delta: float) -> pd.DataFrame:
    """Per-expiry RR/BF at an arbitrary positive ``target_delta`` (call side;
    put side uses ``-target_delta``).
    """
    cols = ["expiry", "tau", "dte", "atm_iv", "put_iv", "call_iv", "rr", "bf"]
    if chain is None or chain.empty:
        return pd.DataFrame(columns=cols)
    atm_by_exp = term_structure(chain).set_index("expiry")["atm_iv"]
    rows = []
    for expiry, grp in chain.groupby("expiry"):
        tau = float(grp["tau"].iloc[0])
        if tau <= 0:
            continue
        valid = grp[grp["iv"].notna() & grp["delta"].notna()]
        calls = valid[(valid["right"] == "C") &
                      (valid["delta"] > 0) & (valid["delta"] < 1)]
        puts = valid[(valid["right"] == "P") &
                     (valid["delta"] < 0) & (valid["delta"] > -1)]
        call_iv = _interp_iv_at_delta(calls[["delta", "iv"]], float(target_delta))
        put_iv = _interp_iv_at_delta(puts[["delta", "iv"]], -float(target_delta))
        atm = float(atm_by_exp.get(expiry, np.nan))
        rr = (call_iv - put_iv
              if np.isfinite(call_iv) and np.isfinite(put_iv) else np.nan)
        if np.isfinite(call_iv) and np.isfinite(put_iv) and np.isfinite(atm):
            bf = 0.5 * (call_iv + put_iv) - atm
        else:
            bf = np.nan
        rows.append({"expiry": expiry, "tau": tau, "dte": tau * 365.25,
                     "atm_iv": atm, "put_iv": put_iv, "call_iv": call_iv,
                     "rr": rr, "bf": bf})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _skew_at_delta_daily(ticker: str, target_delta: float) -> pd.DataFrame:
    """Walk cached chains and compute per-expiry RR/BF at ``target_delta`` per day."""
    key = (f"_skew_at_delta_{round(float(target_delta), 4)}", ticker.upper())
    hit = _DAILY_CACHE.get(key)
    if hit is not None:
        return hit
    rows = []
    for d in _cache.cached_chain_dates(ticker):
        chain = _cache.read_chain(ticker, d)
        if chain is None or chain.empty:
            continue
        try:
            df = _per_expiry_skew_at_delta(chain, target_delta)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df["date"] = pd.Timestamp(d)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _DAILY_CACHE[key] = out
    return out


def _term_one(chain: pd.DataFrame) -> pd.DataFrame:
    return term_structure(chain)


_DAILY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _walk_chains(ticker: str, kind: str, fn) -> pd.DataFrame:
    key = (kind, ticker.upper())
    hit = _DAILY_CACHE.get(key)
    if hit is not None:
        return hit
    rows = []
    for d in _cache.cached_chain_dates(ticker):
        chain = _cache.read_chain(ticker, d)
        if chain is None or chain.empty:
            continue
        try:
            df = fn(chain)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df["date"] = pd.Timestamp(d)
        rows.append(df)
    if not rows:
        out = pd.DataFrame()
    else:
        out = pd.concat(rows, ignore_index=True)
    _DAILY_CACHE[key] = out
    return out


def _interp_at_dte(per_day: pd.DataFrame, target_dte: int, value_col: str) -> pd.Series:
    """For each ``date``, linearly interpolate ``value_col`` at ``target_dte`` over the
    per-expiry rows for that day.
    """
    if per_day.empty or "dte" not in per_day.columns:
        return pd.Series(dtype=float)
    out = {}
    for day, grp in per_day.groupby("date"):
        sub = grp.dropna(subset=["dte", value_col]).sort_values("dte")
        if sub.empty:
            continue
        ds = sub["dte"].to_numpy(dtype=float)
        vs = sub[value_col].to_numpy(dtype=float)
        if len(ds) == 1:
            out[day] = float(vs[0])
            continue
        if target_dte < ds.min() or target_dte > ds.max():
            # nearest-tenor fallback
            idx = int(np.argmin(np.abs(ds - target_dte)))
            out[day] = float(vs[idx])
        else:
            out[day] = float(np.interp(target_dte, ds, vs))
    s = pd.Series(out, dtype=float)
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


class RiskReversalSignal(Signal):
    KIND = "risk_reversal"

    def __init__(self, dte: int = 30, delta: float = 0.25):
        self.dte = int(dte)
        self.delta = float(delta)

    def _compute(self, ticker: str) -> pd.Series:
        per = _skew_at_delta_daily(ticker, self.delta)
        s = _interp_at_dte(per, self.dte, "rr")
        s.name = self.label()
        return _norm_index(s)

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "dte": self.dte, "delta": self.delta}

    @classmethod
    def from_dict(cls, d: dict) -> "RiskReversalSignal":
        return cls(dte=int(d.get("dte", 30)), delta=float(d.get("delta", 0.25)))

    def label(self) -> str:
        d_pct = int(round(self.delta * 100))
        return f"RR_{d_pct}d_{self.dte}"


class ButterflySignal(Signal):
    KIND = "butterfly"

    def __init__(self, dte: int = 30, delta: float = 0.25):
        self.dte = int(dte)
        self.delta = float(delta)

    def _compute(self, ticker: str) -> pd.Series:
        per = _skew_at_delta_daily(ticker, self.delta)
        s = _interp_at_dte(per, self.dte, "bf")
        s.name = self.label()
        return _norm_index(s)

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "dte": self.dte, "delta": self.delta}

    @classmethod
    def from_dict(cls, d: dict) -> "ButterflySignal":
        return cls(dte=int(d.get("dte", 30)), delta=float(d.get("delta", 0.25)))

    def label(self) -> str:
        d_pct = int(round(self.delta * 100))
        return f"BF_{d_pct}d_{self.dte}"


class ForwardVolSignal(Signal):
    """Forward vol between two DTEs.

    For each date, compute the term structure and forward-vol pairs. Then pick
    the consecutive pair whose ``(dte_1, dte_2)`` is closest to the requested
    ``(dte1, dte2)`` and return its ``fwd_vol``. (Snapshot-style rather than
    interpolated since forward-vol is only well-defined between observed
    tenors.)
    """
    KIND = "forward_vol"

    def __init__(self, dte1: int = 30, dte2: int = 60):
        self.dte1 = int(dte1)
        self.dte2 = int(dte2)

    def _compute(self, ticker: str) -> pd.Series:
        per_term = _term_daily(ticker)
        if per_term.empty:
            return pd.Series(dtype=float)
        out = {}
        for day, grp in per_term.groupby("date"):
            ts = grp.drop(columns=["date"])
            fv = forward_vol(ts)
            if fv.empty:
                continue
            tgt1, tgt2 = self.dte1, self.dte2
            score = (fv["dte_1"] - tgt1).abs() + (fv["dte_2"] - tgt2).abs()
            row = fv.iloc[int(score.idxmin())]
            v = float(row["fwd_vol"])
            if np.isfinite(v):
                out[day] = v
        s = pd.Series(out, dtype=float, name=self.label())
        if not s.empty:
            s.index = pd.to_datetime(s.index).normalize()
            s = s.sort_index()
        return s

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "dte1": self.dte1, "dte2": self.dte2}

    @classmethod
    def from_dict(cls, d: dict) -> "ForwardVolSignal":
        return cls(dte1=int(d.get("dte1", 30)), dte2=int(d.get("dte2", 60)))

    def label(self) -> str:
        return f"FWD_VOL_{self.dte1}_{self.dte2}"
