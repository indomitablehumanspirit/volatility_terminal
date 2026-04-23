"""ATM implied-vol term structure."""
from __future__ import annotations

import numpy as np
import pandas as pd


def term_structure(chain: pd.DataFrame) -> pd.DataFrame:
    """ATM-IV per expiry.

    For each expiry, interpolate call and put IV linearly in strike at the
    per-expiry forward ``F`` (already attached to the chain rows), then
    average the two. Returns columns [expiry, tau, forward, call_iv, put_iv,
    atm_iv, n_pts].
    """
    cols = ["expiry", "tau", "forward", "call_iv", "put_iv", "atm_iv", "n_pts"]
    if chain.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for expiry, grp in chain.groupby("expiry"):
        tau = float(grp["tau"].iloc[0])
        F = float(grp["forward"].iloc[0])
        if tau <= 0 or not np.isfinite(F):
            continue
        row = {"expiry": expiry, "tau": tau, "forward": F}
        for right, key in (("C", "call_iv"), ("P", "put_iv")):
            side = grp[(grp["right"] == right) & grp["iv"].notna()] \
                .sort_values("strike")
            if len(side) == 0:
                row[key] = np.nan
                continue
            ks = side["strike"].to_numpy()
            vs = side["iv"].to_numpy()
            if len(ks) >= 2 and ks.min() <= F <= ks.max():
                row[key] = float(np.interp(F, ks, vs))
            else:
                row[key] = float(vs[np.argmin(np.abs(ks - F))])
        both = [v for v in (row["call_iv"], row["put_iv"]) if np.isfinite(v)]
        row["atm_iv"] = float(np.mean(both)) if both else np.nan
        row["n_pts"] = int(grp["iv"].notna().sum())
        out.append(row)
    return pd.DataFrame(out).sort_values("tau").reset_index(drop=True)


def atm_iv_at_dte(chain: pd.DataFrame, target_dte_days: float) -> float:
    """Interpolate ATM IV at ``target_dte_days`` along the term curve."""
    ts = term_structure(chain).dropna(subset=["atm_iv"])
    if ts.empty:
        return float("nan")
    target_tau = target_dte_days / 365.25
    taus = ts["tau"].to_numpy()
    ivs = ts["atm_iv"].to_numpy()
    if target_tau < taus.min() or target_tau > taus.max():
        return float(ivs[int(np.argmin(np.abs(taus - target_tau)))])
    return float(np.interp(target_tau, taus, ivs))
