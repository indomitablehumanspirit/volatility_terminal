"""25-delta risk-reversal and butterfly metrics per expiry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .term import term_structure


_COLS = ["expiry", "tau", "dte", "atm_iv", "put25_iv", "call25_iv",
        "rr_25d", "bf_25d", "n_pts"]


def _interp_iv_at_delta(side: pd.DataFrame, target_delta: float) -> float:
    """Linearly interpolate IV at ``target_delta`` along a single-side (call or put) frame.

    Returns NaN if target is outside the observed delta range (no extrapolation).
    """
    side = side.dropna(subset=["delta", "iv"])
    if len(side) < 2:
        return float("nan")
    s = side.sort_values("delta")
    ds = s["delta"].to_numpy()
    vs = s["iv"].to_numpy()
    if target_delta < ds.min() or target_delta > ds.max():
        return float("nan")
    return float(np.interp(target_delta, ds, vs))


def delta_skew_metrics(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry 25-delta risk reversal and butterfly.

    Columns: expiry, tau, dte, atm_iv, put25_iv, call25_iv, rr_25d, bf_25d, n_pts.
    IV values are in decimal (0.20 = 20%). RR and BF are IV differences in decimal.
    """
    if chain is None or chain.empty:
        return pd.DataFrame(columns=_COLS)

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

        call25 = _interp_iv_at_delta(calls[["delta", "iv"]], 0.25)
        put25 = _interp_iv_at_delta(puts[["delta", "iv"]], -0.25)
        atm = float(atm_by_exp.get(expiry, np.nan))

        rr = call25 - put25 if np.isfinite(call25) and np.isfinite(put25) else np.nan
        if np.isfinite(call25) and np.isfinite(put25) and np.isfinite(atm):
            bf = 0.5 * (call25 + put25) - atm
        else:
            bf = np.nan

        rows.append({
            "expiry": expiry,
            "tau": tau,
            "dte": tau * 365.25,
            "atm_iv": atm,
            "put25_iv": put25,
            "call25_iv": call25,
            "rr_25d": rr,
            "bf_25d": bf,
            "n_pts": int(len(valid)),
        })

    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
