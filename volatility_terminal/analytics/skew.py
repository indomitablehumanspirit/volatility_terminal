"""OTM-only IV skew curves by expiry."""
from __future__ import annotations

import numpy as np
import pandas as pd


def skew_for_expiry(chain: pd.DataFrame, expiry) -> pd.DataFrame:
    """OTM-only (puts for K<F, calls for K>=F) IV curve for one expiry."""
    cols = ["expiry", "tau", "forward", "strike", "right", "iv", "log_moneyness"]
    if chain.empty:
        return pd.DataFrame(columns=cols)
    g = chain[(chain["expiry"] == expiry) & chain["iv"].notna()].copy()
    if g.empty:
        return pd.DataFrame(columns=cols)
    F = float(g["forward"].iloc[0])
    otm = g[((g["right"] == "P") & (g["strike"] < F)) |
            ((g["right"] == "C") & (g["strike"] >= F))].copy()
    otm["log_moneyness"] = np.log(otm["strike"] / F)
    return (otm[cols].sort_values("log_moneyness").reset_index(drop=True))


def all_skew_curves(chain: pd.DataFrame) -> dict:
    """Map expiry_timestamp -> OTM skew DataFrame for every expiry in chain."""
    if chain.empty:
        return {}
    return {exp: skew_for_expiry(chain, exp)
            for exp in chain["expiry"].drop_duplicates().tolist()}
