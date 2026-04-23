"""Historical trade simulation engine.

Simulates a multi-leg options position day-by-day from an entry date through
the last leg's expiry using cached chain data and BS repricing for missing days.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Callable

import numpy as np
import pandas as pd

from ..data import cache
from ..pricing.bs import bs_price, greeks
from ..pricing.rates import RateCurve


@dataclass
class Leg:
    symbol: str
    right: str           # "C" or "P"
    strike: float
    expiry: pd.Timestamp # midnight UTC
    qty: int             # signed: + long, - short
    entry_price: float   # mid at entry ($/contract, not × 100)


@dataclass
class HedgeConfig:
    use_interval: bool = False
    interval_days: int = 5
    use_delta_threshold: bool = False
    delta_threshold: float = 100.0   # shares of net delta
    use_spot_move: bool = False
    spot_move_pct: float = 2.0


@dataclass
class HedgeTrade:
    date: date
    trigger: str                  # "interval" | "delta" | "spot_move"
    shares_traded: int            # signed: + bought, - sold
    spot: float
    portfolio_delta_before: float


@dataclass
class DailyState:
    date: date
    spot: float
    pnl: float
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    portfolio_theta: float


@dataclass
class SimResult:
    states: list[DailyState]
    hedge_log: list[HedgeTrade]
    entry_greeks: dict
    final_pnl: float
    entry_date: date
    ticker: str
    legs: list[Leg]
    error: str | None = None


def run_simulation(
    ticker: str,
    entry_date: date,
    legs: list[Leg],
    stock_shares: int,
    hedge_config: HedgeConfig,
    rates: RateCurve,
    progress_cb: Callable | None = None,
) -> SimResult:
    """Simulate a multi-leg options position from entry_date through last expiry."""

    _err = lambda msg: SimResult(
        states=[], hedge_log=[], entry_greeks={}, final_pnl=0.0,
        entry_date=entry_date, ticker=ticker, legs=legs, error=msg,
    )

    # --- Phase 0: Validate ---
    cached_dates = set(cache.cached_chain_dates(ticker))
    if entry_date not in cached_dates:
        return _err(f"Entry date {entry_date} not in cache for {ticker}.")

    entry_chain = cache.read_chain(ticker, entry_date)
    if entry_chain is None or entry_chain.empty:
        return _err(f"Could not read chain for {ticker} on {entry_date}.")

    for leg in legs:
        rows = entry_chain[entry_chain["symbol"] == leg.symbol]
        if rows.empty:
            return _err(f"Symbol {leg.symbol} not found in entry chain.")
        if not np.isfinite(rows.iloc[0]["iv"]):
            return _err(f"No valid IV for {leg.symbol} on entry date.")

    underlying = cache.read_underlying(ticker)
    if underlying is None or underlying.empty:
        return _err(f"No underlying price data for {ticker}.")

    # Build date-indexed close series (tz-naive date index)
    und = underlying.copy()
    und["timestamp"] = pd.to_datetime(und["timestamp"]).dt.tz_localize(None).dt.normalize()
    und = und.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    close_series = und["close"]

    # --- Determine trading calendar ---
    last_expiry = max(leg.expiry.date() for leg in legs)
    mask = (close_series.index.date >= entry_date) & (close_series.index.date <= last_expiry)
    trading_days = [ts.date() for ts in close_series.index[mask]]
    if not trading_days:
        return _err("No trading days found between entry date and last expiry.")

    def get_spot(sim_day: date) -> float:
        ts = pd.Timestamp(sim_day).tz_localize(None)
        if ts in close_series.index:
            return float(close_series[ts])
        # forward-fill: use last available close
        prior = close_series[close_series.index <= ts]
        if prior.empty:
            return float(close_series.iloc[0])
        return float(prior.iloc[-1])

    entry_spot = get_spot(entry_date)

    # --- Phase 1: Initialize carry-forward state ---
    last_iv: dict[str, float] = {}
    last_q: dict[pd.Timestamp, float] = {}

    for leg in legs:
        row = entry_chain[entry_chain["symbol"] == leg.symbol].iloc[0]
        last_iv[leg.symbol] = float(row["iv"])
        last_q[leg.expiry] = float(row["q"]) if np.isfinite(row["q"]) else 0.0

    # PnL accounting: initial_cash = -(cost of entering all legs)
    initial_cash = (
        -sum(leg.qty * leg.entry_price * 100 for leg in legs)
        - stock_shares * entry_spot
    )
    cash = initial_cash
    shares_held = stock_shares

    last_hedge_spot = entry_spot
    last_hedge_day_idx = 0

    # Compute entry greeks
    entry_greeks = _sum_greeks(legs, entry_chain, stock_shares)

    # --- Phase 2: Main simulation loop ---
    open_legs: list[Leg] = list(legs)
    states: list[DailyState] = []
    hedge_log: list[HedgeTrade] = []
    total = len(trading_days)

    for i, sim_day in enumerate(trading_days):
        spot = get_spot(sim_day)

        # Step A: Settle expired legs (before repricing)
        still_open = []
        for leg in open_legs:
            if sim_day >= leg.expiry.date():
                intrinsic = bs_price(spot, leg.strike, 0, 0, 0.001, leg.right, 0)
                cash += leg.qty * intrinsic * 100
            else:
                still_open.append(leg)
        open_legs = still_open

        if not open_legs:
            # All legs settled — record final states with no options component
            pnl = cash + shares_held * spot
            port_delta = float(shares_held)
            states.append(DailyState(
                date=sim_day, spot=spot, pnl=pnl,
                portfolio_delta=port_delta,
                portfolio_gamma=0.0, portfolio_vega=0.0, portfolio_theta=0.0,
            ))
            if progress_cb:
                progress_cb(i + 1, total, f"{ticker} {sim_day}")
            continue

        # Step B+C: Reprice open legs
        prices_today: dict[str, float] = {}
        greeks_today: dict[str, dict] = {}

        day_chain = cache.read_chain(ticker, sim_day)
        if day_chain is not None and not day_chain.empty:
            for leg in open_legs:
                rows = day_chain[day_chain["symbol"] == leg.symbol]
                if not rows.empty:
                    row = rows.iloc[0]
                    prices_today[leg.symbol] = float(row["mid"])
                    greeks_today[leg.symbol] = {
                        "delta": float(row["delta"]) if np.isfinite(row["delta"]) else 0.0,
                        "gamma": float(row["gamma"]) if np.isfinite(row["gamma"]) else 0.0,
                        "vega": float(row["vega"]) if np.isfinite(row["vega"]) else 0.0,
                        "theta": float(row["theta"]) if np.isfinite(row["theta"]) else 0.0,
                    }
                    if np.isfinite(row["iv"]):
                        last_iv[leg.symbol] = float(row["iv"])
                    if np.isfinite(row["q"]):
                        last_q[leg.expiry] = float(row["q"])
                else:
                    price, g = _bs_reprice(leg, spot, sim_day, last_iv, last_q, rates)
                    prices_today[leg.symbol] = price
                    greeks_today[leg.symbol] = g
        else:
            for leg in open_legs:
                price, g = _bs_reprice(leg, spot, sim_day, last_iv, last_q, rates)
                prices_today[leg.symbol] = price
                greeks_today[leg.symbol] = g

        # Step D: Portfolio delta before hedge
        port_delta = sum(
            leg.qty * greeks_today[leg.symbol]["delta"] * 100
            for leg in open_legs
        ) + shares_held
        port_gamma = sum(
            leg.qty * greeks_today[leg.symbol].get("gamma", 0) * 100
            for leg in open_legs
        )
        port_vega = sum(
            leg.qty * greeks_today[leg.symbol].get("vega", 0) * 100
            for leg in open_legs
        )
        port_theta = sum(
            leg.qty * greeks_today[leg.symbol].get("theta", 0) * 100
            for leg in open_legs
        )

        # Step E: Check hedge triggers (not on day 0)
        if i > 0 and open_legs:
            days_since_hedge = i - last_hedge_day_idx
            spot_move_pct = abs(spot - last_hedge_spot) / last_hedge_spot * 100 if last_hedge_spot else 0.0

            trigger_name = None
            if hedge_config.use_interval and days_since_hedge >= hedge_config.interval_days:
                trigger_name = "interval"
            if hedge_config.use_delta_threshold and abs(port_delta) > hedge_config.delta_threshold:
                trigger_name = trigger_name or "delta"
            if hedge_config.use_spot_move and spot_move_pct >= hedge_config.spot_move_pct:
                trigger_name = trigger_name or "spot_move"

            if trigger_name:
                shares_to_trade = -int(round(port_delta))
                delta_before = port_delta
                cash -= shares_to_trade * spot
                shares_held += shares_to_trade
                port_delta += shares_to_trade
                last_hedge_spot = spot
                last_hedge_day_idx = i
                hedge_log.append(HedgeTrade(
                    date=sim_day,
                    trigger=trigger_name,
                    shares_traded=shares_to_trade,
                    spot=spot,
                    portfolio_delta_before=delta_before,
                ))

        # Step F: PnL
        option_value = sum(leg.qty * prices_today[leg.symbol] * 100 for leg in open_legs)
        pnl = cash + option_value + shares_held * spot

        states.append(DailyState(
            date=sim_day, spot=spot, pnl=pnl,
            portfolio_delta=port_delta,
            portfolio_gamma=port_gamma,
            portfolio_vega=port_vega,
            portfolio_theta=port_theta,
        ))

        if progress_cb:
            progress_cb(i + 1, total, f"{ticker} {sim_day}")

    final_pnl = states[-1].pnl if states else 0.0
    return SimResult(
        states=states,
        hedge_log=hedge_log,
        entry_greeks=entry_greeks,
        final_pnl=final_pnl,
        entry_date=entry_date,
        ticker=ticker,
        legs=legs,
    )


def _sum_greeks(legs: list[Leg], chain: pd.DataFrame, stock_shares: int) -> dict:
    delta = float(stock_shares)
    gamma = vega = theta = 0.0
    for leg in legs:
        rows = chain[chain["symbol"] == leg.symbol]
        if rows.empty:
            continue
        row = rows.iloc[0]
        mult = leg.qty * 100
        delta += mult * (float(row["delta"]) if np.isfinite(row["delta"]) else 0.0)
        gamma += mult * (float(row["gamma"]) if np.isfinite(row["gamma"]) else 0.0)
        vega += mult * (float(row["vega"]) if np.isfinite(row["vega"]) else 0.0)
        theta += mult * (float(row["theta"]) if np.isfinite(row["theta"]) else 0.0)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def _bs_reprice(
    leg: Leg,
    spot: float,
    sim_day: date,
    last_iv: dict[str, float],
    last_q: dict[pd.Timestamp, float],
    rates: RateCurve,
) -> tuple[float, dict]:
    expiry_ts = leg.expiry + pd.Timedelta(hours=20)
    as_of_ts = pd.Timestamp(datetime.combine(sim_day, time(20, 0)), tz="UTC")
    tau = max((expiry_ts - as_of_ts).total_seconds() / (365.25 * 86400.0), 0.0)

    sigma = last_iv.get(leg.symbol, 0.2)
    q = last_q.get(leg.expiry, 0.0)
    r = rates.r_at(sim_day, tau) if tau > 0 else 0.0

    price = bs_price(spot, leg.strike, tau, r, sigma, leg.right, q)
    g = greeks(spot, leg.strike, tau, r, sigma, leg.right, q)
    # greeks() returns nan for tau <= 0; normalize to 0
    g_clean = {k: float(v) if np.isfinite(v) else 0.0 for k, v in g.items()}
    return price, g_clean
