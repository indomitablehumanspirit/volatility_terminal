"""Earnings move analytics: realized vs implied, one row per historical event.

Realized move definition:
    BMO (before market open): close_{E-1} -> close_{E}
    AMC (after market close): close_{E}   -> close_{E+1}
    UNKNOWN: treated as AMC.
Reported as the absolute log return.

Implied move (two methods):
    (1) Front-expiry ATM straddle divided by spot.
    (2) Term-structure earnings-jump decomposition: assume a flat diffusion
        volatility sigma_d and an earnings-day jump J shared by two expiries
        bracketing the event. Solve the 2x2 system:
            sigma_front^2 * T_f = sigma_d^2 * T_f + J^2
            sigma_back^2  * T_b = sigma_d^2 * T_b + J^2
        =>  sigma_d^2 = (sigma_back^2 * T_b - sigma_front^2 * T_f) / (T_b - T_f)
            J^2       = sigma_front^2 * T_f - sigma_d^2 * T_f
        Implied jump move = sqrt(J^2) in log-return units.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .term import term_structure


def _prev_business_day(d: date) -> date:
    """Previous business day (skip Sat/Sun — ignores holidays)."""
    out = d - timedelta(days=1)
    while out.weekday() >= 5:
        out -= timedelta(days=1)
    return out


def realized_move(bars: pd.DataFrame, earnings_date: date,
                  time_of_day: str) -> float:
    """Absolute log-return close-before → close-after earnings.

    ``bars`` must have columns [timestamp, close] covering the event window.
    Returns NaN if either close is missing.
    """
    if bars is None or bars.empty:
        return float("nan")
    b = bars.copy()
    b["_date"] = pd.to_datetime(b["timestamp"]).dt.tz_localize(None).dt.date
    b = b.drop_duplicates("_date").set_index("_date")["close"].astype(float)

    tod = (time_of_day or "UNKNOWN").upper()
    if tod == "BMO":
        # Pre-earnings close = last trading day strictly before E.
        # Post-earnings close = E itself (if market traded) else next trading day.
        prior = b[b.index < earnings_date]
        after = b[b.index >= earnings_date]
    else:  # AMC or UNKNOWN
        prior = b[b.index <= earnings_date]
        after = b[b.index > earnings_date]

    if prior.empty or after.empty:
        return float("nan")
    p0 = float(prior.iloc[-1])
    p1 = float(after.iloc[0])
    if p0 <= 0 or p1 <= 0:
        return float("nan")
    return abs(float(np.log(p1 / p0)))


def _pick_atm_row(side: pd.DataFrame, forward: float) -> Optional[pd.Series]:
    """Row nearest forward strike for a single right ('C'/'P') filtered frame."""
    if side.empty or not np.isfinite(forward):
        return None
    idx = (side["strike"] - forward).abs().idxmin()
    return side.loc[idx]


def implied_move_straddle(chain: pd.DataFrame, earnings_date: date) -> float:
    """ATM straddle on the first expiry strictly after earnings_date, / spot.

    Returns log-return-equivalent magnitude (i.e. straddle/spot, which is the
    standard desk convention and is directly comparable to |log(P1/P0)|).
    """
    if chain is None or chain.empty:
        return float("nan")
    c = chain.copy()
    c["_exp"] = pd.to_datetime(c["expiry"]).dt.tz_localize(None).dt.date
    future = c[c["_exp"] > earnings_date]
    if future.empty:
        return float("nan")
    front_exp = future["_exp"].min()
    ce = future[future["_exp"] == front_exp]
    F = float(ce["forward"].iloc[0])
    spot = float(ce["spot"].iloc[0])
    if not np.isfinite(F) or not np.isfinite(spot) or spot <= 0:
        return float("nan")
    call = _pick_atm_row(ce[ce["right"] == "C"], F)
    put = _pick_atm_row(ce[ce["right"] == "P"], F)
    if call is None or put is None:
        return float("nan")
    straddle = float(call["mid"]) + float(put["mid"])
    if not np.isfinite(straddle) or straddle <= 0:
        return float("nan")
    return straddle / spot


def implied_move_term_decomp(chain: pd.DataFrame, earnings_date: date) -> float:
    """Earnings-jump magnitude from front/back ATM IV decomposition.

    Uses the first two expiries strictly after earnings_date. Returns NaN when
    the decomposition would be degenerate (sigma_d^2 < 0, J^2 < 0, or only one
    expiry available).
    """
    if chain is None or chain.empty:
        return float("nan")
    ts = term_structure(chain).dropna(subset=["atm_iv", "tau"])
    if ts.empty:
        return float("nan")
    ts = ts.copy()
    ts["_exp"] = pd.to_datetime(ts["expiry"]).dt.tz_localize(None).dt.date
    future = ts[ts["_exp"] > earnings_date].sort_values("tau")
    if len(future) < 2:
        return float("nan")
    front = future.iloc[0]
    back = future.iloc[1]
    sf2, tf = float(front["atm_iv"]) ** 2, float(front["tau"])
    sb2, tb = float(back["atm_iv"]) ** 2, float(back["tau"])
    if tb <= tf:
        return float("nan")
    sigma_d2 = (sb2 * tb - sf2 * tf) / (tb - tf)
    if sigma_d2 < 0:
        # negative base variance -> front expiry unusually cheap vs back;
        # fall back to treating all front variance as jump
        sigma_d2 = 0.0
    j2 = sf2 * tf - sigma_d2 * tf
    if j2 <= 0:
        return float("nan")
    return float(np.sqrt(j2))


def _chain_as_of(earnings_date: date, time_of_day: str) -> date:
    """The trading day whose EOD chain we want to snapshot for implied move.

    BMO: the last trading day strictly before earnings.
    AMC/UNKNOWN: the earnings day itself (close captures the pre-print IV).
    """
    tod = (time_of_day or "UNKNOWN").upper()
    if tod == "BMO":
        return _prev_business_day(earnings_date)
    d = earnings_date
    while d.weekday() >= 5:
        d = _prev_business_day(d)
    return d


def build_earnings_table(
    ticker: str,
    bars: pd.DataFrame,
    earnings: pd.DataFrame,
    chain_loader: Callable[[str, date], Optional[pd.DataFrame]],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Assemble per-earnings-event rows with realized + implied move metrics.

    ``chain_loader(ticker, day) -> DataFrame | None`` is expected to only hit
    local cache (no network) so this can run across many events quickly.
    Missing chain data (e.g. pre-Feb-2024 for Alpaca) yields NaN implied cols.
    """
    ticker = ticker.upper()
    if earnings is None or earnings.empty:
        return pd.DataFrame(columns=[
            "date", "time_of_day", "implied_straddle", "implied_jump",
            "realized", "spread",
        ])

    rows = []
    total = len(earnings)
    for i, r in enumerate(earnings.itertuples(index=False)):
        d = r.date
        tod = r.time_of_day
        rm = realized_move(bars, d, tod)

        chain_day = _chain_as_of(d, tod)
        try:
            chain = chain_loader(ticker, chain_day)
        except Exception:
            chain = None
        im_s = implied_move_straddle(chain, d) if chain is not None else float("nan")
        im_j = implied_move_term_decomp(chain, d) if chain is not None else float("nan")

        spread = rm - im_s if np.isfinite(rm) and np.isfinite(im_s) else float("nan")
        rows.append({
            "date": d,
            "time_of_day": tod,
            "implied_straddle": im_s,
            "implied_jump": im_j,
            "realized": rm,
            "spread": spread,
        })
        if progress_cb:
            progress_cb(i + 1, total, str(d))

    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
