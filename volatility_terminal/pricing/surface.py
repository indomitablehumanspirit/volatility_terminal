"""Build a regular IV surface grid from a raw chain snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def build_surface(chain: pd.DataFrame,
                  n_k: int = 41, n_t: int = 21,
                  k_range: tuple[float, float] = (-0.35, 0.35)) -> dict:
    """Interpolate IV onto a regular (log-moneyness x tau) grid for plotting.

    Filters to OTM quotes (puts below forward, calls above) to avoid
    put-call arbitrage noise near the money. Returns a dict with:
        'k_grid'  : 1-D array of log-moneyness values (length n_k)
        'tau_grid': 1-D array of tau values in years (length n_t)
        'iv_grid' : 2-D array shape (n_k, n_t), IV in decimal, NaN where no data
        'raw'     : DataFrame [log_moneyness, tau, iv, right, strike, expiry]
    """
    if chain.empty:
        return {"k_grid": np.array([]), "tau_grid": np.array([]),
                "iv_grid": np.zeros((0, 0)), "raw": pd.DataFrame()}
    g = chain.dropna(subset=["iv", "forward", "tau"]).copy()
    g = g[g["iv"] > 0]
    if g.empty:
        return {"k_grid": np.array([]), "tau_grid": np.array([]),
                "iv_grid": np.zeros((0, 0)), "raw": pd.DataFrame()}
    g["log_moneyness"] = np.log(g["strike"] / g["forward"])
    otm = g[((g["right"] == "P") & (g["strike"] <  g["forward"])) |
            ((g["right"] == "C") & (g["strike"] >= g["forward"]))]
    if otm.empty:
        otm = g
    otm = otm[(otm["log_moneyness"] >= k_range[0]) &
              (otm["log_moneyness"] <= k_range[1])]
    if len(otm) < 4:
        return {"k_grid": np.array([]), "tau_grid": np.array([]),
                "iv_grid": np.zeros((0, 0)), "raw": otm}

    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    tau_min = max(otm["tau"].min(), 1 / 365.25)
    tau_max = otm["tau"].max()
    tau_grid = np.linspace(tau_min, tau_max, n_t)
    KK, TT = np.meshgrid(k_grid, tau_grid, indexing="ij")
    pts = otm[["log_moneyness", "tau"]].to_numpy()
    vals = otm["iv"].to_numpy()
    iv_grid = griddata(pts, vals, (KK, TT), method="linear")
    return {
        "k_grid": k_grid,
        "tau_grid": tau_grid,
        "iv_grid": iv_grid,
        "raw": otm[["log_moneyness", "tau", "iv", "right", "strike", "expiry"]].copy(),
    }
