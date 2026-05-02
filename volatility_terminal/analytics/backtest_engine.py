"""Generic backtest engine driven by composable signal rules + structures.

Generalization of ``straddle_backtest.run_straddle_backtest``: same day-loop
shape, but entry/exit triggers come from ``Rule`` objects evaluated up front,
and the leg builder is ``structures.build_legs`` (any structure kind).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from ..data import cache
from ..pricing.rates import RateCurve
from .backtest_config import BacktestConfig
from .signals.base import rule_from_config
from .simulation import Leg, _bs_reprice
from .structures import build_legs


@dataclass
class HedgeEvent:
    date: date
    trigger: str
    shares_traded: int
    spot: float
    portfolio_delta_before: float


@dataclass
class ClosedTrade:
    entry_date: date
    exit_date: date
    expiry: date
    strikes: tuple[float, ...]
    rights: tuple[str, ...]
    sides: tuple[str, ...]      # "long"/"short" per leg
    credit: float               # net cash flow at entry (positive for credit)
    pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame   # columns: date, daily_pnl, cum_pnl
    trades: list[ClosedTrade]
    hedge_events: list[HedgeEvent]
    summary: dict
    ticker: str
    config: BacktestConfig
    error: str | None = None


# ---------------------------------------------------------------------------

def _price_and_greeks(leg: Leg, day: date, chain: pd.DataFrame | None,
                      spot: float, last_iv: dict, last_q: dict,
                      rates: RateCurve) -> tuple[float, dict]:
    if chain is not None and not chain.empty:
        rows = chain[chain["symbol"] == leg.symbol]
        if not rows.empty:
            row = rows.iloc[0]
            mid = float(row["mid"])
            if np.isfinite(mid):
                g = {
                    "delta": float(row["delta"]) if np.isfinite(row["delta"]) else 0.0,
                    "gamma": float(row["gamma"]) if np.isfinite(row["gamma"]) else 0.0,
                    "vega": float(row["vega"]) if np.isfinite(row["vega"]) else 0.0,
                    "theta": float(row["theta"]) if np.isfinite(row["theta"]) else 0.0,
                }
                if np.isfinite(row["iv"]):
                    last_iv[leg.symbol] = float(row["iv"])
                if np.isfinite(row["q"]):
                    last_q[leg.expiry] = float(row["q"])
                return mid, g
    return _bs_reprice(leg, spot, day, last_iv, last_q, rates)


# ---------------------------------------------------------------------------

def run_backtest(
    ticker: str,
    config: BacktestConfig,
    rates: RateCurve,
    progress_cb: Callable | None = None,
) -> BacktestResult:
    ticker = ticker.upper()

    def _err(msg: str) -> BacktestResult:
        return BacktestResult(
            equity_curve=pd.DataFrame(columns=["date", "daily_pnl", "cum_pnl"]),
            trades=[], hedge_events=[], summary={},
            ticker=ticker, config=config, error=msg,
        )

    chain_dates = cache.cached_chain_dates(ticker)
    if not chain_dates:
        return _err(f"No cached chains for {ticker}. Run backfill first.")

    if config.start is not None:
        chain_dates = [d for d in chain_dates if d >= config.start]
    if config.end is not None:
        chain_dates = [d for d in chain_dates if d <= config.end]
    if not chain_dates:
        return _err("No cached chain dates inside the requested range.")

    under = cache.read_underlying(ticker)
    if under is None or under.empty:
        return _err(f"No underlying price data for {ticker}.")
    und = under.copy()
    und["date"] = pd.to_datetime(und["timestamp"]).dt.tz_localize(None).dt.normalize()
    und = und.drop_duplicates("date").set_index("date").sort_index()
    close_series = und["close"].astype(float)

    def get_spot(d: date) -> float:
        ts = pd.Timestamp(d)
        if ts in close_series.index:
            return float(close_series[ts])
        prior = close_series[close_series.index <= ts]
        if prior.empty:
            return float(close_series.iloc[0])
        return float(prior.iloc[-1])

    # ---- Pre-compute entry / exit rule masks once over the full date range ----
    date_index = pd.DatetimeIndex([pd.Timestamp(d) for d in chain_dates])

    entry_mask = pd.Series(True, index=date_index)
    if not config.entry_rule.is_empty():
        rule = rule_from_config(config.entry_rule)
        m = rule.evaluate(ticker)
        entry_mask = m.reindex(date_index).fillna(False).astype(bool)

    exit_mask = pd.Series(False, index=date_index)
    if not config.exit_rule.is_empty():
        rule = rule_from_config(config.exit_rule)
        m = rule.evaluate(ticker)
        exit_mask = m.reindex(date_index).fillna(False).astype(bool)

    # ---- state ----
    realized_pnl: float = 0.0
    trades: list[ClosedTrade] = []
    hedge_events: list[HedgeEvent] = []
    equity_rows: list[dict] = []
    prev_equity: float = 0.0

    open_legs: list[Leg] | None = None
    credit: float = 0.0
    entry_day: date | None = None
    last_iv: dict[str, float] = {}
    last_q: dict[pd.Timestamp, float] = {}
    hedge_cash: float = 0.0
    shares_held: int = 0
    last_hedge_spot: float = 0.0
    last_hedge_day_idx: int = 0
    entry_day_idx: int = 0

    prev_entry_signal_value: bool = False  # for edge_only re-arm

    total = len(chain_dates)

    def _open_trade(i: int, day: date, chain: pd.DataFrame) -> bool:
        nonlocal open_legs, credit, entry_day, last_iv, last_q
        nonlocal hedge_cash, shares_held, last_hedge_spot, last_hedge_day_idx, entry_day_idx
        legs = build_legs(config.structure, chain, day)
        if legs is None:
            return False
        spot = get_spot(day)
        # Cash flow at entry: paying for longs (negative cash), receiving for shorts (positive)
        credit_now = -sum(leg.qty * leg.entry_price * 100 for leg in legs)
        open_legs = legs
        credit = credit_now
        entry_day = day
        entry_day_idx = i
        for leg in legs:
            row = chain[chain["symbol"] == leg.symbol].iloc[0]
            if np.isfinite(row["iv"]):
                last_iv[leg.symbol] = float(row["iv"])
            if np.isfinite(row["q"]):
                last_q[leg.expiry] = float(row["q"])
        hedge_cash = 0.0
        shares_held = 0
        last_hedge_spot = spot
        last_hedge_day_idx = i
        return True

    def _close_trade(reason: str, day: date, spot: float,
                     leg_val_total: float) -> None:
        nonlocal realized_pnl, open_legs, credit, entry_day, hedge_cash, shares_held
        option_pnl = credit + leg_val_total
        hedge_pnl = hedge_cash + shares_held * spot
        trade_pnl = option_pnl + hedge_pnl
        realized_pnl += trade_pnl

        strikes = tuple(leg.strike for leg in open_legs)
        rights = tuple(leg.right for leg in open_legs)
        sides = tuple("long" if leg.qty > 0 else "short" for leg in open_legs)
        expiry_date = max(leg.expiry.date() for leg in open_legs)
        trades.append(ClosedTrade(
            entry_date=entry_day, exit_date=day, expiry=expiry_date,
            strikes=strikes, rights=rights, sides=sides,
            credit=credit, pnl=trade_pnl, exit_reason=reason,
        ))
        open_legs = None
        credit = 0.0
        entry_day = None
        hedge_cash = 0.0
        shares_held = 0

    # ---------------- main loop ----------------
    for i, day in enumerate(chain_dates):
        if progress_cb and (i % 5 == 0 or i == total - 1):
            progress_cb(i + 1, total, f"{ticker} {day}")

        spot = get_spot(day)
        ts = pd.Timestamp(day)
        entry_signal = bool(entry_mask.get(ts, False))
        exit_signal = bool(exit_mask.get(ts, False))

        # If position open: mark-to-market, hedge, maybe close
        closed_this_bar = False
        day_chain = cache.read_chain(ticker, day)
        if open_legs is not None:
            leg_val_total = 0.0
            port_delta = float(shares_held)
            any_expired = False
            for leg in open_legs:
                if day >= leg.expiry.date():
                    intrinsic = max(0.0, (spot - leg.strike) if leg.right == "C"
                                    else (leg.strike - spot))
                    leg_val_total += leg.qty * intrinsic * 100
                    any_expired = True
                else:
                    price, g = _price_and_greeks(
                        leg, day, day_chain, spot, last_iv, last_q, rates,
                    )
                    leg_val_total += leg.qty * price * 100
                    port_delta += leg.qty * g.get("delta", 0.0) * 100

            open_pnl = credit + leg_val_total
            net_pnl = open_pnl + hedge_cash + shares_held * spot

            # Delta hedge (skip on entry day)
            if not any_expired and i > entry_day_idx and config.hedge is not None:
                hc = config.hedge
                days_since = i - last_hedge_day_idx
                spot_move_pct = (abs(spot - last_hedge_spot) / last_hedge_spot * 100.0
                                 if last_hedge_spot else 0.0)
                trigger = None
                if hc.use_interval and days_since >= hc.interval_days:
                    trigger = "interval"
                if hc.use_delta_threshold and abs(port_delta) > hc.delta_threshold:
                    trigger = trigger or "delta"
                if hc.use_spot_move and spot_move_pct >= hc.spot_move_pct:
                    trigger = trigger or "spot_move"
                if trigger is not None:
                    shares_to_trade = -int(round(port_delta))
                    if shares_to_trade != 0:
                        hedge_cash -= shares_to_trade * spot
                        shares_held += shares_to_trade
                        hedge_events.append(HedgeEvent(
                            date=day, trigger=trigger,
                            shares_traded=shares_to_trade, spot=spot,
                            portfolio_delta_before=port_delta,
                        ))
                    last_hedge_spot = spot
                    last_hedge_day_idx = i

            # Exit evaluation
            reason: str | None = None
            if any_expired:
                reason = "expiration"
            elif exit_signal:
                reason = "exit_signal"
            elif config.use_dte_exit:
                min_dte = min(max((leg.expiry.date() - day).days, 0)
                              for leg in open_legs)
                if min_dte <= config.dte_exit_threshold:
                    reason = "dte_touch"
            if reason is None and config.use_stop_loss:
                if net_pnl <= -config.stop_loss_pct / 100.0 * abs(credit):
                    reason = "stop_loss"
            if reason is None and config.use_profit_target:
                if net_pnl >= config.profit_target_pct / 100.0 * abs(credit):
                    reason = "profit_target"

            if reason is not None:
                _close_trade(reason, day, spot, leg_val_total)
                closed_this_bar = True

        # If flat (or just closed): try to open next.
        # rearm:
        #  - any_bar : entry_signal must be True today
        #  - edge_only : entry_signal must be True today AND was False yesterday
        if open_legs is None and not closed_this_bar:
            ok_to_arm = entry_signal
            if config.rearm == "edge_only":
                ok_to_arm = ok_to_arm and (not prev_entry_signal_value)
            if ok_to_arm:
                if day_chain is not None and not day_chain.empty:
                    _open_trade(i, day, day_chain)

        # Equity snapshot
        if open_legs is not None:
            mtm = 0.0
            for leg in open_legs:
                if day >= leg.expiry.date():
                    intrinsic = max(0.0, (spot - leg.strike) if leg.right == "C"
                                    else (leg.strike - spot))
                    mtm += leg.qty * intrinsic * 100
                else:
                    price, _ = _price_and_greeks(
                        leg, day, day_chain, spot, last_iv, last_q, rates,
                    )
                    mtm += leg.qty * price * 100
            unrealized = credit + mtm + hedge_cash + shares_held * spot
            equity = realized_pnl + unrealized
        else:
            equity = realized_pnl

        daily = equity - prev_equity
        prev_equity = equity
        equity_rows.append({"date": pd.Timestamp(day),
                            "daily_pnl": daily, "cum_pnl": equity})

        prev_entry_signal_value = entry_signal

    equity_df = pd.DataFrame(equity_rows)

    # ---------------- summary ----------------
    n_trades = len(trades)
    pnls = np.array([t.pnl for t in trades]) if trades else np.array([])
    summary: dict = {
        "total_pnl": float(realized_pnl),
        "n_trades": n_trades,
        "win_rate": float((pnls > 0).mean()) if n_trades else 0.0,
        "avg_trade_pnl": float(pnls.mean()) if n_trades else 0.0,
        "avg_hold_days": float(np.mean([(t.exit_date - t.entry_date).days for t in trades]))
            if n_trades else 0.0,
        "n_hedges": len(hedge_events),
    }
    if not equity_df.empty:
        cum = equity_df["cum_pnl"].values
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        summary["max_drawdown"] = float(dd.min())
        daily_pnl = equity_df["daily_pnl"].values
        std = float(np.std(daily_pnl))
        summary["sharpe"] = (float(np.mean(daily_pnl) / std * np.sqrt(252))
                             if std > 0 else 0.0)
    else:
        summary["max_drawdown"] = 0.0
        summary["sharpe"] = 0.0

    return BacktestResult(
        equity_curve=equity_df,
        trades=trades,
        hedge_events=hedge_events,
        summary=summary,
        ticker=ticker,
        config=config,
    )
