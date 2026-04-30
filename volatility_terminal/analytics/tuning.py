"""Hyperparameter tuning for the generic backtester.

Random search over user-selected parameters with a strict in-sample /
out-of-sample split: the optimizer only ever evaluates on IS dates; OOS is a
true holdout used once at the best-found point.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from ..data import cache
from ..pricing.rates import RateCurve
from .backtest_config import BacktestConfig
from .backtest_engine import BacktestResult, run_backtest


@dataclass
class TuningParam:
    name: str
    label: str
    min_val: float
    max_val: float
    is_integer: bool


@dataclass
class TuningConfig:
    params: list[TuningParam]
    n_trials: int
    split_pct: float
    objective: str
    sensitivity_points: int = 25
    min_trades: int = 5


@dataclass
class TuningResult:
    best_config: BacktestConfig
    best_is_score: float
    oos_score: float
    all_trials: list[tuple[dict, float]]
    sensitivity: dict[str, tuple[np.ndarray, np.ndarray]]
    oos_sensitivity_point: dict[str, float]
    is_result: BacktestResult
    oos_result: BacktestResult
    is_dates: tuple[date, date]
    oos_dates: tuple[date, date]
    objective: str
    error: str | None = None


# ---------------------------------------------------------------------------

def compute_objective(result: BacktestResult, objective: str,
                      min_trades: int) -> float:
    if result.error or not result.trades:
        return -np.inf
    if len(result.trades) < min_trades:
        return -np.inf
    eq = result.equity_curve
    if eq.empty:
        return -np.inf
    daily = eq["daily_pnl"].values
    if objective == "profit_factor":
        wins = daily[daily > 0].sum()
        losses = -daily[daily < 0].sum()
        if losses == 0:
            return np.inf if wins > 0 else 0.0
        return float(wins / losses)
    if objective == "sharpe":
        return float(result.summary.get("sharpe", -np.inf))
    if objective == "total_pnl":
        return float(result.summary.get("total_pnl", -np.inf))
    return -np.inf


# ---------------------------------------------------------------------------

def apply_params(base_config: BacktestConfig,
                 param_values: dict[str, float]) -> BacktestConfig:
    cfg = copy.deepcopy(base_config)
    for name, val in param_values.items():
        if name.startswith("leg_") and name.endswith("_dte"):
            i = int(name.split("_")[1])
            if 0 <= i < len(cfg.structure.legs):
                cfg.structure.legs[i].dte = int(round(val))
        elif name.startswith("leg_") and name.endswith("_delta"):
            i = int(name.split("_")[1])
            if 0 <= i < len(cfg.structure.legs):
                if cfg.structure.legs[i].delta_target is not None:
                    cfg.structure.legs[i].delta_target = float(val)
        elif name == "profit_target_pct":
            cfg.profit_target_pct = float(val)
        elif name == "stop_loss_pct":
            cfg.stop_loss_pct = float(val)
        elif name == "dte_exit_threshold":
            cfg.dte_exit_threshold = int(round(val))
        elif name.startswith("entry_cond_") and name.endswith("_threshold"):
            i = int(name.split("_")[2])
            if 0 <= i < len(cfg.entry_rule.conditions):
                cfg.entry_rule.conditions[i].threshold = float(val)
        elif name.startswith("exit_cond_") and name.endswith("_threshold"):
            i = int(name.split("_")[2])
            if 0 <= i < len(cfg.exit_rule.conditions):
                cfg.exit_rule.conditions[i].threshold = float(val)
        elif name == "hedge_interval_days":
            cfg.hedge.interval_days = int(round(val))
        elif name == "hedge_delta_threshold":
            cfg.hedge.delta_threshold = float(val)
        elif name == "hedge_spot_move_pct":
            cfg.hedge.spot_move_pct = float(val)
    return cfg


# ---------------------------------------------------------------------------

def _sample(p: TuningParam) -> float:
    v = random.uniform(p.min_val, p.max_val)
    if p.is_integer:
        v = int(round(v))
    return v


def _err_result(msg: str, base_config: BacktestConfig,
                objective: str) -> TuningResult:
    empty = BacktestResult(
        equity_curve=pd.DataFrame(columns=["date", "daily_pnl", "cum_pnl"]),
        trades=[], hedge_events=[], summary={},
        ticker="", config=base_config, error=msg,
    )
    return TuningResult(
        best_config=base_config, best_is_score=float("-inf"),
        oos_score=float("-inf"), all_trials=[], sensitivity={},
        oos_sensitivity_point={}, is_result=empty, oos_result=empty,
        is_dates=(date.min, date.min), oos_dates=(date.min, date.min),
        objective=objective, error=msg,
    )


def run_tuning(
    ticker: str,
    base_config: BacktestConfig,
    tuning_config: TuningConfig,
    rates: RateCurve,
    progress_cb: Callable | None = None,
) -> TuningResult:
    ticker = ticker.upper()
    obj = tuning_config.objective
    min_trades = tuning_config.min_trades

    if not tuning_config.params:
        return _err_result("No parameters selected for tuning.", base_config, obj)

    chain_dates = cache.cached_chain_dates(ticker)
    if not chain_dates:
        return _err_result(f"No cached chains for {ticker}.", base_config, obj)
    if base_config.start is not None:
        chain_dates = [d for d in chain_dates if d >= base_config.start]
    if base_config.end is not None:
        chain_dates = [d for d in chain_dates if d <= base_config.end]
    if len(chain_dates) < 20:
        return _err_result(
            f"Need at least 20 cached dates in range; got {len(chain_dates)}.",
            base_config, obj)

    split_idx = int(len(chain_dates) * tuning_config.split_pct)
    split_idx = max(10, min(split_idx, len(chain_dates) - 5))
    is_start = chain_dates[0]
    is_end = chain_dates[split_idx - 1]
    oos_start = chain_dates[split_idx]
    oos_end = chain_dates[-1]

    # ---- Random search on IS ----
    best_score = float("-inf")
    best_params: dict = {}
    all_trials: list[tuple[dict, float]] = []
    n = tuning_config.n_trials
    for t in range(n):
        sample = {p.name: _sample(p) for p in tuning_config.params}
        trial_cfg = apply_params(base_config, sample)
        trial_cfg.start = is_start
        trial_cfg.end = is_end
        try:
            res = run_backtest(ticker, trial_cfg, rates)
            score = compute_objective(res, obj, min_trades)
        except Exception:
            score = float("-inf")
        all_trials.append((sample, score))
        if score > best_score:
            best_score = score
            best_params = sample
        if progress_cb and (t % 5 == 0 or t == n - 1):
            progress_cb(t + 1, n,
                        f"random search • best={best_score:.3f}")

    if not best_params:
        return _err_result("No valid trials produced a score.", base_config, obj)

    # ---- Best on IS (full result for display) ----
    best_cfg_is = apply_params(base_config, best_params)
    best_cfg_is.start = is_start
    best_cfg_is.end = is_end
    is_result = run_backtest(ticker, best_cfg_is, rates)
    best_is_score = compute_objective(is_result, obj, min_trades)

    # ---- OOS evaluation at best params ----
    best_cfg_oos = apply_params(base_config, best_params)
    best_cfg_oos.start = oos_start
    best_cfg_oos.end = oos_end
    oos_result = run_backtest(ticker, best_cfg_oos, rates)
    oos_score = compute_objective(oos_result, obj, min_trades)

    # ---- Sensitivity sweeps (per-param, IS only) ----
    sensitivity: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    oos_pt: dict[str, float] = {}
    n_pts = max(3, tuning_config.sensitivity_points)
    total_sweep = len(tuning_config.params) * n_pts
    swept = 0
    for p in tuning_config.params:
        xs = np.linspace(p.min_val, p.max_val, n_pts)
        if p.is_integer:
            xs = np.unique(np.round(xs).astype(int))
        ys = []
        for v in xs:
            sample = dict(best_params)
            sample[p.name] = float(v)
            cfg = apply_params(base_config, sample)
            cfg.start = is_start
            cfg.end = is_end
            try:
                res = run_backtest(ticker, cfg, rates)
                s = compute_objective(res, obj, min_trades)
            except Exception:
                s = float("-inf")
            ys.append(s)
            swept += 1
            if progress_cb and (swept % 5 == 0):
                progress_cb(swept, total_sweep,
                            f"sensitivity • {p.label}")
        sensitivity[p.name] = (np.asarray(xs, dtype=float),
                                np.asarray(ys, dtype=float))
        oos_pt[p.name] = oos_score

    return TuningResult(
        best_config=best_cfg_is,
        best_is_score=best_is_score,
        oos_score=oos_score,
        all_trials=all_trials,
        sensitivity=sensitivity,
        oos_sensitivity_point=oos_pt,
        is_result=is_result,
        oos_result=oos_result,
        is_dates=(is_start, is_end),
        oos_dates=(oos_start, oos_end),
        objective=obj,
    )
