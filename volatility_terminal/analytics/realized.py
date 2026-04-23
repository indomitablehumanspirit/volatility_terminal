"""Realized-vol estimators from daily underlying OHLC."""
from __future__ import annotations

import numpy as np
import pandas as pd


def close_to_close(daily_close: pd.Series, window: int = 30) -> pd.Series:
    """Annualized rolling std of daily log returns."""
    log_ret = np.log(daily_close / daily_close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def parkinson(high: pd.Series, low: pd.Series, window: int = 30) -> pd.Series:
    """Parkinson (1980) high-low volatility estimator, annualized."""
    hl = np.log(high / low) ** 2
    var = hl.rolling(window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * 252)
