"""Rolling short straddle / strangle backtester.

Walks forward through cached chain dates selling one short volatility position
at a time. Reuses ``HedgeConfig`` semantics from ``simulation.py`` for delta
hedging and the same BS fallback for days where a leg is missing from the
cached chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal

import numpy as np
import pandas as pd

from ..data import cache
from .realized import close_to_close
from .iv_timeseries import build_iv_timeseries
from .vrp import compute_vrp
from .simulation import HedgeConfig, Leg, _bs_reprice, _price_and_greeks
from ..pricing.rates import RateCurve


Structure = Literal["straddle", "strangle"]
IVRankMethod = Literal["rank", "percentile"]


@dataclass
class StraddleBacktestConfig:
    structure: Structure = "straddle"
    target_dte: int = 30
    qty: int = 1                       # contracts per leg (always entered short)
    strangle_delta: float = 0.25       # |Δ| target for strangle wings

    # Entry filters
    use_iv_rank: bool = False
    iv_rank_threshold: float = 50.0
    iv_rank_lookback: int = 252
    iv_rank_method: IVRankMethod = "rank"
    use_vrp_filter: bool = False
    vrp_threshold: float = 0.02
    conditional_only: bool = False     # if True, wait for filter to pass on roll

    # Exit rules
    use_dte_exit: bool = False
    dte_exit_threshold: int = 21
    use_profit_target: bool = False
    profit_target_pct: float = 50.0    # % of credit
    use_stop_loss: bool = False
    stop_loss_pct: float = 200.0       # % of credit

    hedge: HedgeConfig = field(default_factory=HedgeConfig)


@dataclass
class ClosedTrade:
    entry_date: date
    exit_date: date
    expiry: date
    strikes: tuple[float, ...]
    rights: tuple[str, ...]
    credit: float           # $, net credit received at entry (positive for short)
    pnl: float              # $, realized PnL including any hedge PnL on this trade
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame   # columns: date, daily_pnl, cum_pnl
    trades: list[ClosedTrade]
    summary: dict
    ticker: str
    config: StraddleBacktestConfig
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers

def _compute_iv_rank_series(iv: pd.Series, lookback: int, method: IVRankMethod) -> pd.Series:
    """Rolling IV rank (0-100) keyed by the same index as ``iv``."""
    if method == "percentile":
        return iv.rolling(lookback, min_periods=max(20, lookback // 4)).apply(
            lambda w: (np.sum(w <= w[-1]) / len(w)) * 100.0, raw=True,
        )
    # default: classic tastytrade rank
    def _rank(w):
        lo, hi = np.nanmin(w), np.nanmax(w)
        if not np.isfinite(hi - lo) or hi - lo == 0:
            return np.nan
        return (w[-1] - lo) / (hi - lo) * 100.0

    return iv.rolling(lookback, min_periods=max(20, lookback // 4)).apply(_rank, raw=True)


def _pick_expiry(chain: pd.DataFrame, target_dte: int) -> pd.Timestamp | None:
    exp = chain.dropna(subset=["tau"]).copy()
    exp = exp[exp["tau"] > 0]
    if exp.empty:
        return None
    by_exp = exp.groupby("expiry")["tau"].first()
    target_tau = target_dte / 365.25
    diff = (by_exp - target_tau).abs()
    return diff.idxmin()


def _pick_legs(chain: pd.DataFrame, expiry: pd.Timestamp, cfg: StraddleBacktestConfig,
               entry_date: date) -> list[Leg] | None:
    sub = chain[chain["expiry"] == expiry]
    if sub.empty:
        return None

    def _find(right: str, target_delta_abs: float | None) -> pd.Series | None:
        side = sub[(sub["right"] == right) & sub["mid"].notna() & sub["delta"].notna()].copy()
        if side.empty:
            return None
        if target_delta_abs is None:
            # ATM = strike closest to forward
            F = float(side["forward"].iloc[0]) if np.isfinite(side["forward"].iloc[0]) else float(side["spot"].iloc[0])
            idx = (side["strike"] - F).abs().idxmin()
        else:
            idx = (side["delta"].abs() - target_delta_abs).abs().idxmin()
        return side.loc[idx]

    if cfg.structure == "straddle":
        calls = sub[(sub["right"] == "C") & sub["mid"].notna()]
        puts = sub[(sub["right"] == "P") & sub["mid"].notna()]
        common = sorted(set(calls["strike"]).intersection(puts["strike"]))
        if not common:
            return None
        ref_row = calls.iloc[0]
        F = float(ref_row["forward"]) if np.isfinite(ref_row["forward"]) else float(ref_row["spot"])
        strike = min(common, key=lambda k: abs(k - F))
        call_row = calls[calls["strike"] == strike].iloc[0]
        put_row = puts[puts["strike"] == strike].iloc[0]
    else:  # strangle
        call_row = _find("C", cfg.strangle_delta)
        put_row = _find("P", cfg.strangle_delta)
        if call_row is None or put_row is None:
            return None

    legs = []
    for row in (call_row, put_row):
        mid = float(row["mid"])
        if not np.isfinite(mid) or mid <= 0:
            return None
        legs.append(Leg(
            symbol=str(row["symbol"]),
            right=str(row["right"]),
            strike=float(row["strike"]),
            expiry=pd.Timestamp(row["expiry"]),
            qty=-abs(cfg.qty),  # always short
            entry_price=mid,
        ))
    return legs


# ---------------------------------------------------------------------------
# Main entry point

def run_straddle_backtest(
    ticker: str,
    config: StraddleBacktestConfig,
    rates: RateCurve,
    progress_cb: Callable | None = None,
) -> BacktestResult:
    ticker = ticker.upper()

    def _err(msg: str) -> BacktestResult:
        return BacktestResult(
            equity_curve=pd.DataFrame(columns=["date", "daily_pnl", "cum_pnl"]),
            trades=[], summary={}, ticker=ticker, config=config, error=msg,
        )

    chain_dates = cache.cached_chain_dates(ticker)
    if not chain_dates:
        return _err(f"No cached chains for {ticker}. Run backfill first.")

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

    # Precompute IV rank + VRP series if filters enabled
    iv_rank_by_date: dict[date, float] = {}
    vrp_by_date: dict[date, float] = {}
    if config.use_iv_rank or config.use_vrp_filter:
        try:
            iv_ts = build_iv_timeseries(ticker, config.target_dte)
        except Exception as e:
            return _err(f"Could not build IV timeseries: {e}")
        iv_ts = iv_ts.copy()
        _dts = pd.to_datetime(iv_ts["date"])
        if _dts.dt.tz is not None:
            _dts = _dts.dt.tz_convert(None)
        iv_ts["date_key"] = _dts.dt.date
        if config.use_iv_rank:
            ranks = _compute_iv_rank_series(
                iv_ts["atm_iv"], config.iv_rank_lookback, config.iv_rank_method,
            )
            iv_rank_by_date = dict(zip(iv_ts["date_key"], ranks.values))
        if config.use_vrp_filter:
            rv = close_to_close(close_series, window=config.target_dte)
            merged = compute_vrp(iv_ts[["date", "atm_iv", "spot"]], rv)
            _mdts = pd.to_datetime(merged["date"])
            if _mdts.dt.tz is not None:
                _mdts = _mdts.dt.tz_convert(None)
            merged["date_key"] = _mdts.dt.date
            vrp_by_date = dict(zip(merged["date_key"], merged["vrp"].values))

    # ---------------- state ----------------
    realized_pnl: float = 0.0
    trades: list[ClosedTrade] = []
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

    total = len(chain_dates)

    def filter_passes(d: date) -> bool:
        # Filters only gate entry when conditional-only is on.
        # When off, auto-roll every day regardless of filter checkboxes.
        if not config.conditional_only:
            return True
        if config.use_iv_rank:
            r = iv_rank_by_date.get(d, np.nan)
            if not np.isfinite(r) or r <= config.iv_rank_threshold:
                return False
        if config.use_vrp_filter:
            v = vrp_by_date.get(d, np.nan)
            if not np.isfinite(v) or v <= config.vrp_threshold:
                return False
        return True

    def close_trade(reason: str, i: int, day: date, spot: float,
                    leg_val_total: float) -> None:
        nonlocal realized_pnl, open_legs, credit, entry_day, hedge_cash, shares_held
        # Option realized = credit + leg_val_total (leg_val = qty × price × 100,
        # negative for open shorts). On exit at current mid/intrinsic, position
        # closes and credit + leg_val = realized option pnl for this trade.
        option_pnl = credit + leg_val_total
        # Flatten stock at current spot
        hedge_pnl = hedge_cash + shares_held * spot
        trade_pnl = option_pnl + hedge_pnl
        realized_pnl += trade_pnl

        strikes = tuple(leg.strike for leg in open_legs)
        rights = tuple(leg.right for leg in open_legs)
        expiry_date = max(leg.expiry.date() for leg in open_legs)
        trades.append(ClosedTrade(
            entry_date=entry_day, exit_date=day, expiry=expiry_date,
            strikes=strikes, rights=rights,
            credit=credit, pnl=trade_pnl, exit_reason=reason,
        ))
        open_legs = None
        credit = 0.0
        entry_day = None
        hedge_cash = 0.0
        shares_held = 0

    def open_trade(i: int, day: date, chain: pd.DataFrame) -> bool:
        nonlocal open_legs, credit, entry_day, last_iv, last_q, \
            hedge_cash, shares_held, last_hedge_spot, last_hedge_day_idx, entry_day_idx
        expiry = _pick_expiry(chain, config.target_dte)
        if expiry is None:
            return False
        legs = _pick_legs(chain, expiry, config, day)
        if legs is None:
            return False
        spot = get_spot(day)
        open_legs = legs
        credit = -sum(leg.qty * leg.entry_price * 100 for leg in legs)  # short → positive
        entry_day = day
        entry_day_idx = i
        # seed carry-forward state from chain
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

    # ---------------- main loop ----------------
    for i, day in enumerate(chain_dates):
        if progress_cb and (i % 5 == 0 or i == total - 1):
            progress_cb(i + 1, total, f"{ticker} {day}")

        spot = get_spot(day)

        # If position open, mark-to-market, hedge, maybe close
        closed_this_bar = False
        day_chain = cache.read_chain(ticker, day)
        if open_legs is not None:
            # Reprice & greeks, settle expired intrinsically
            leg_val_total = 0.0
            port_delta = float(shares_held)
            any_expired = False
            for leg in open_legs:
                if day >= leg.expiry.date():
                    # settle at intrinsic
                    intrinsic = max(0.0, (spot - leg.strike) if leg.right == "C" else (leg.strike - spot))
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

            # Delta hedge (skip on entry day itself)
            if not any_expired and i > entry_day_idx and config.hedge is not None:
                hc = config.hedge
                days_since = i - last_hedge_day_idx
                spot_move_pct = (abs(spot - last_hedge_spot) / last_hedge_spot * 100.0
                                 if last_hedge_spot else 0.0)
                fire = False
                if hc.use_interval and days_since >= hc.interval_days:
                    fire = True
                if hc.use_delta_threshold and abs(port_delta) > hc.delta_threshold:
                    fire = True
                if hc.use_spot_move and spot_move_pct >= hc.spot_move_pct:
                    fire = True
                if fire:
                    shares_to_trade = -int(round(port_delta))
                    if shares_to_trade != 0:
                        hedge_cash -= shares_to_trade * spot
                        shares_held += shares_to_trade
                    last_hedge_spot = spot
                    last_hedge_day_idx = i

            # Exit evaluation
            reason: str | None = None
            if any_expired:
                reason = "expiration"
            elif config.use_dte_exit:
                # min DTE across open legs
                min_dte = min(max((leg.expiry.date() - day).days, 0) for leg in open_legs)
                if min_dte <= config.dte_exit_threshold:
                    reason = "dte_touch"
            if reason is None and config.use_stop_loss:
                if net_pnl <= -config.stop_loss_pct / 100.0 * abs(credit):
                    reason = "stop_loss"
            if reason is None and config.use_profit_target:
                if net_pnl >= config.profit_target_pct / 100.0 * abs(credit):
                    reason = "profit_target"

            if reason is not None:
                close_trade(reason, i, day, spot, leg_val_total)
                closed_this_bar = True

        # If flat (either already, or just closed): try to open next.
        # Conditional-only: skip entry this day if filter fails.
        # Else if filter enabled (non-conditional): filter must still pass to open.
        # Else (no filter): open unconditionally.
        if open_legs is None and not closed_this_bar:
            # Only try entering if filters pass (or none set)
            if filter_passes(day):
                if day_chain is not None and not day_chain.empty:
                    open_trade(i, day, day_chain)
        elif open_legs is None and closed_this_bar:
            # Just closed today — re-entry happens next trading day, never same bar.
            pass

        # Equity snapshot
        if open_legs is not None:
            # still open after all the above
            # recompute mark for snapshot consistency
            # (leg_val_total from above if not closed, else position was reset)
            mtm = 0.0
            for leg in open_legs:
                if day >= leg.expiry.date():
                    intrinsic = max(0.0, (spot - leg.strike) if leg.right == "C" else (leg.strike - spot))
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
        equity_rows.append({"date": pd.Timestamp(day), "daily_pnl": daily, "cum_pnl": equity})

    equity_df = pd.DataFrame(equity_rows)

    # ---------------- summary ----------------
    n_trades = len(trades)
    pnls = np.array([t.pnl for t in trades]) if trades else np.array([])
    summary = {
        "total_pnl": float(realized_pnl),
        "n_trades": n_trades,
        "win_rate": float((pnls > 0).mean()) if n_trades else 0.0,
        "avg_trade_pnl": float(pnls.mean()) if n_trades else 0.0,
        "avg_hold_days": float(np.mean([(t.exit_date - t.entry_date).days for t in trades]))
            if n_trades else 0.0,
    }
    if not equity_df.empty:
        cum = equity_df["cum_pnl"].values
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        summary["max_drawdown"] = float(dd.min())
        daily_pnl = equity_df["daily_pnl"].values
        std = float(np.std(daily_pnl))
        summary["sharpe"] = float(np.mean(daily_pnl) / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        summary["max_drawdown"] = 0.0
        summary["sharpe"] = 0.0

    return BacktestResult(
        equity_curve=equity_df,
        trades=trades,
        summary=summary,
        ticker=ticker,
        config=config,
    )
