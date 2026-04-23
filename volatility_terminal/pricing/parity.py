"""Infer forward price F and dividend yield q from put-call parity.

Parity:  C - P = S * exp(-q*tau) - K * exp(-r*tau)

Given an observed (C, P) pair at strike K with known (S, r, tau), solve for q.
For each expiry we use strikes closest to the spot and average the result
for stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def infer_forward_and_q(chain: pd.DataFrame, S: float, r_at_tau,
                         n_strikes: int = 5) -> pd.DataFrame:
    """Per-expiry forward F and dividend yield q from put-call parity.

    Parameters
    ----------
    chain : long-format DataFrame with columns [expiry, tau, strike, right, mid]
    S : spot price of the underlying
    r_at_tau : callable(tau_years) -> decimal rate
    n_strikes : number of strikes closest to spot to average

    Returns DataFrame indexed by expiry with columns [tau, r, forward, q].
    Rows with insufficient call/put pairs are dropped.
    """
    if chain.empty:
        return pd.DataFrame(columns=["tau", "r", "forward", "q"])

    calls = chain[chain["right"] == "C"][["expiry", "tau", "strike", "mid"]] \
        .rename(columns={"mid": "c_mid"})
    puts  = chain[chain["right"] == "P"][["expiry", "tau", "strike", "mid"]] \
        .rename(columns={"mid": "p_mid"})
    pairs = calls.merge(puts, on=["expiry", "tau", "strike"], how="inner")
    pairs = pairs.dropna(subset=["c_mid", "p_mid"])
    if pairs.empty:
        return pd.DataFrame(columns=["tau", "r", "forward", "q"])

    out = []
    for (expiry, tau), grp in pairs.groupby(["expiry", "tau"]):
        if tau <= 0:
            continue
        r = float(r_at_tau(tau))
        near = grp.iloc[(grp["strike"] - S).abs().argsort().values[:n_strikes]].copy()
        rhs_term = near["c_mid"] - near["p_mid"] + near["strike"] * np.exp(-r * tau)
        rhs_term = rhs_term[rhs_term > 0]
        if rhs_term.empty:
            q = 0.0
        else:
            q_samples = -np.log(rhs_term / S) / tau
            q_samples = q_samples[np.isfinite(q_samples)]
            q = float(np.median(q_samples)) if len(q_samples) else 0.0
            q = float(np.clip(q, -0.05, 0.20))
        F = S * np.exp((r - q) * tau)
        out.append({"expiry": expiry, "tau": tau, "r": r, "forward": F, "q": q})

    return (pd.DataFrame(out)
            .set_index("expiry")
            .sort_values("tau"))
