"""Black-Scholes pricing, greeks, and implied-volatility solver.

Ported verbatim from the reference project's ``option_chain.py``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S, K, tau, r, sigma, right, q=0.0):
    if tau <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if right == "C" else max(K - S, 0.0)
        return intrinsic
    sqrt_t = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if right == "C":
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)


def greeks(S, K, tau, r, sigma, right, q=0.0):
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan,
                    theta=np.nan, rho=np.nan)
    sqrt_t = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf = norm.pdf(d1)
    disc_r = np.exp(-r * tau)
    disc_q = np.exp(-q * tau)
    gamma_ = disc_q * pdf / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf * sqrt_t / 100.0
    if right == "C":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sqrt_t)
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1)) / 365.0
        rho = K * tau * disc_r * norm.cdf(d2) / 100.0
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sqrt_t)
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1)) / 365.0
        rho = -K * tau * disc_r * norm.cdf(-d2) / 100.0
    return dict(delta=delta, gamma=gamma_, vega=vega, theta=theta, rho=rho)


def implied_vol(price, S, K, tau, r, right, q=0.0):
    if price is None or not np.isfinite(price) or price <= 0 or tau <= 0:
        return np.nan
    intrinsic = (max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0) if right == "C"
                 else max(K * np.exp(-r * tau) - S * np.exp(-q * tau), 0.0))
    upper = S * np.exp(-q * tau) if right == "C" else K * np.exp(-r * tau)
    if price < intrinsic - 1e-6 or price > upper + 1e-6:
        return np.nan
    f = lambda s: bs_price(S, K, tau, r, s, right, q) - price
    try:
        return brentq(f, 1e-4, 5.0, maxiter=100, xtol=1e-6)
    except ValueError:
        return np.nan
