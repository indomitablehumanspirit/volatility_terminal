"""Forward implied-variance / vol between consecutive term-structure tenors."""
from __future__ import annotations

import numpy as np
import pandas as pd


_COLS = ["expiry_1", "expiry_2", "dte_1", "dte_2",
        "iv_1", "iv_2", "fwd_var", "fwd_vol"]


def forward_vol(ts: pd.DataFrame) -> pd.DataFrame:
    """Forward vol between consecutive expiries of a term-structure frame.

    ``ts`` is the output of ``analytics.term.term_structure`` (sorted by tau,
    with ``atm_iv`` in decimal). Rows with NaN ATM IV are dropped first so
    pairs always span two valid tenors.

    For each consecutive pair (T1, T2):
        var_fwd = (iv2**2 * t2 - iv1**2 * t1) / (t2 - t1)
        fwd_vol = sqrt(var_fwd)  if var_fwd >= 0 else NaN
    Negative ``fwd_var`` signals calendar-spread arbitrage.
    """
    if ts is None or ts.empty:
        return pd.DataFrame(columns=_COLS)
    clean = ts.dropna(subset=["atm_iv"]).sort_values("tau").reset_index(drop=True)
    if len(clean) < 2:
        return pd.DataFrame(columns=_COLS)

    rows = []
    for i in range(len(clean) - 1):
        a = clean.iloc[i]
        b = clean.iloc[i + 1]
        t1, t2 = float(a["tau"]), float(b["tau"])
        if t2 <= t1:
            continue
        iv1, iv2 = float(a["atm_iv"]), float(b["atm_iv"])
        fwd_var = (iv2 * iv2 * t2 - iv1 * iv1 * t1) / (t2 - t1)
        fwd_vol_val = float(np.sqrt(fwd_var)) if fwd_var >= 0 else float("nan")
        rows.append({
            "expiry_1": a["expiry"],
            "expiry_2": b["expiry"],
            "dte_1": t1 * 365.25,
            "dte_2": t2 * 365.25,
            "iv_1": iv1,
            "iv_2": iv2,
            "fwd_var": fwd_var,
            "fwd_vol": fwd_vol_val,
        })

    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows)
