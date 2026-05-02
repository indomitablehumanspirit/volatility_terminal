"""Orchestrate chain fetch: Alpaca contracts + bars -> enriched DataFrame.

Enrichment:
    - parse OCC symbol -> (expiry, right, strike)
    - tau = (expiry - as_of) / 365.25 years
    - mid = close price (daily bar)
    - spot = underlying close on `day`
    - forward F, dividend q per expiry from put-call parity
    - IV and greeks via Black-Scholes
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache
from .alpaca_client import AlpacaOptionsData
from ..pricing import bs, parity
from ..pricing.occ import parse_occ
from ..pricing.rates import RateCurve


class ChainFetcher:
    def __init__(self, alpaca: AlpacaOptionsData, rate_curve: RateCurve):
        self.alpaca = alpaca
        self.rates = rate_curve

    def get_chain(self, ticker: str, day: date,
                  use_cache: bool = True) -> pd.DataFrame:
        """Return an enriched chain DataFrame for (ticker, day).

        Columns: symbol, expiry, right, strike, tau, close, volume, mid, spot,
                 r, q, forward, iv, delta, gamma, vega, theta, rho.
        """
        ticker = ticker.upper()
        if use_cache:
            cached = cache.read_chain(ticker, day)
            if cached is not None and len(cached):
                return cached

        spot = self._get_spot(ticker, day)
        if not np.isfinite(spot):
            return pd.DataFrame()

        contracts = self.alpaca.list_contracts(ticker, as_of=day,
                                               expiration_after=day)
        if contracts.empty:
            return pd.DataFrame()

        bars = self.alpaca.get_daily_bars(contracts["symbol"].tolist(), day)
        if bars.empty:
            return pd.DataFrame()

        out = self._enrich_day(contracts, bars, spot, day)
        if not out.empty:
            cache.write_chain(ticker, day, out)
        return out

    def backfill_range(self, ticker: str, start: date, end: date,
                       progress_cb=None) -> int:
        """Bulk-fetch and cache chains for every uncached trading day in [start, end].

        Fetches the contract universe and bars once for the full range instead of
        repeating those calls per day, drastically reducing API round-trips.
        Returns the number of days written to cache.
        """
        ticker = ticker.upper()
        days = list(pd.bdate_range(start, end).date)
        cached = set(cache.cached_chain_dates(ticker))
        days_needed = [d for d in days if d not in cached]
        if not days_needed:
            return 0
        total = len(days_needed)

        def _status(msg: str) -> None:
            if progress_cb:
                progress_cb(0, total, msg)

        # Fetch the full underlying price history for the range at once.
        _status(f"Fetching {ticker} underlying price history...")
        under = self.alpaca.get_daily_stock_bars(ticker, start, end)
        if under is None or under.empty:
            return 0
        cache.write_underlying(ticker, under)
        under["_date"] = pd.to_datetime(under["timestamp"]).dt.date

        # Fetch only contracts that could have bars in [start, end].
        # Cap expiration at end + 2 years: contracts expiring further out are
        # overwhelmingly un-traded during a historical backfill window.
        exp_cap = end + timedelta(days=730)
        _status(f"Fetching {ticker} contract universe...")
        all_contracts = self.alpaca.list_contracts(
            ticker, as_of=start, expiration_after=start,
            expiration_before=exp_cap,
            status_cb=_status,
        )
        if all_contracts.empty:
            return 0
        # Drop contracts that had already expired before the first day we need,
        # then limit to symbols that could possibly have traded during the range.
        all_contracts = all_contracts[
            all_contracts["expiration"].dt.date >= start
        ].copy()

        # Fetch all bars for all symbols across the full date range in one pass.
        _status(f"Fetched {len(all_contracts)} contracts, fetching historical bars...")
        all_bars = self.alpaca.get_bars_range(
            all_contracts["symbol"].tolist(), start, end,
            status_cb=_status,
        )
        if all_bars.empty:
            return 0
        all_bars["_date"] = pd.to_datetime(all_bars["timestamp"]).dt.date

        written = 0
        for i, day in enumerate(days_needed):
            day_bars = all_bars[all_bars["_date"] == day].copy()
            if day_bars.empty:
                if progress_cb:
                    progress_cb(i + 1, total, f"{ticker} {day} (no bars)")
                continue

            day_symbols = set(day_bars["symbol"])
            day_contracts = all_contracts[
                (all_contracts["expiration"].dt.date >= day) &
                (all_contracts["symbol"].isin(day_symbols))
            ].copy()

            prior = under[under["_date"] <= day].sort_values("_date")
            if prior.empty:
                continue
            spot = float(prior["close"].iloc[-1])
            if not np.isfinite(spot):
                continue

            out = self._enrich_day(day_contracts, day_bars, spot, day)
            if not out.empty:
                cache.write_chain(ticker, day, out)
                written += 1

            if progress_cb:
                progress_cb(i + 1, total, f"{ticker} {day}")

        return written

    def _enrich_day(self, contracts: pd.DataFrame, bars: pd.DataFrame,
                    spot: float, day: date) -> pd.DataFrame:
        """Merge contracts+bars and compute IV/greeks. Returns enriched DataFrame."""
        df = contracts.merge(bars[["symbol", "close", "volume"]], on="symbol",
                             how="inner")
        df = df[df["close"] > 0].copy()
        df["mid"] = df["close"]  # EOD close as proxy for mid

        # parse OCC to normalize expiry timestamp (UTC 20:00 on expiry date)
        parsed = df["symbol"].map(parse_occ)
        df["expiry_ts"] = [p[0] for p in parsed]
        # fall back to contract expiration field if parsing failed
        exp_from_field = pd.to_datetime(df["expiration"]).dt.tz_localize("UTC") \
            + pd.Timedelta(hours=20)
        df["expiry_ts"] = df["expiry_ts"].fillna(exp_from_field)
        df["expiry"] = df["expiry_ts"].dt.normalize()

        as_of_ts = pd.Timestamp(datetime.combine(day, time(20, 0), tzinfo=timezone.utc))
        df["tau"] = (df["expiry_ts"] - as_of_ts).dt.total_seconds() / (365.25 * 86400.0)
        df = df[df["tau"] > 0].copy()
        df["spot"] = spot

        r_of_tau = lambda tau: self.rates.r_at(day, tau)
        per_exp = parity.infer_forward_and_q(df, spot, r_of_tau)
        if per_exp.empty:
            return pd.DataFrame()
        per_exp_df = per_exp[["r", "forward", "q"]].reset_index()
        # align expiry dtypes: both must be tz-aware at midnight
        df["_exp_key"] = pd.to_datetime(df["expiry"]).dt.tz_convert("UTC").dt.normalize()
        per_exp_df["_exp_key"] = pd.to_datetime(per_exp_df["expiry"]).dt.tz_convert("UTC").dt.normalize()
        df = df.merge(per_exp_df[["_exp_key", "r", "forward", "q"]],
                      on="_exp_key", how="left").drop(columns=["_exp_key"])
        df = df.dropna(subset=["r", "forward", "q"])

        df["iv"] = [
            bs.implied_vol(mid, s, k, t, r, right, q)
            for mid, s, k, t, r, right, q
            in zip(df["mid"], df["spot"], df["strike"],
                   df["tau"], df["r"], df["right"], df["q"])
        ]

        g = bs.greeks_vec(
            df["spot"].values, df["strike"].values, df["tau"].values,
            df["r"].values, df["iv"].values, df["right"].values,
            df["q"].values,
        )
        df["delta"] = g["delta"]
        df["gamma"] = g["gamma"]
        df["vega"] = g["vega"]
        df["theta"] = g["theta"]
        df["rho"] = g["rho"]

        out_cols = ["symbol", "expiry", "expiry_ts", "right", "strike", "tau",
                    "close", "volume", "mid", "spot", "r", "q", "forward",
                    "iv", "delta", "gamma", "vega", "theta", "rho"]
        return df[out_cols].sort_values(["expiry", "strike", "right"]).reset_index(drop=True)

    def _get_spot(self, ticker: str, day: date) -> float:
        """Close price of the underlying on ``day`` (backfilled from cache)."""
        under = cache.read_underlying(ticker)
        need_fetch = under is None or len(under) == 0
        if not need_fetch:
            under["date"] = pd.to_datetime(under["timestamp"]).dt.date
            if day not in set(under["date"].tolist()):
                need_fetch = True
        if need_fetch:
            # fetch a wide window so subsequent queries hit cache
            start = pd.Timestamp(day) - pd.Timedelta(days=30)
            end = pd.Timestamp(day) + pd.Timedelta(days=1)
            under = self.alpaca.get_daily_stock_bars(ticker, start.date(), end.date())
            if under is None or under.empty:
                return float("nan")
            cache.write_underlying(ticker, under)
            under["date"] = pd.to_datetime(under["timestamp"]).dt.date
        # last close at/before day
        prior = under[under["date"] <= day].sort_values("date")
        if prior.empty:
            return float("nan")
        return float(prior["close"].iloc[-1])
