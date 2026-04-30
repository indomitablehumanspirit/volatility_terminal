"""Generic multi-leg structure builder.

Given a ``StructureParams`` describing a free-form list of legs (each with its
own right, side, DTE, delta target, and quantity), ``build_legs`` picks
concrete contracts off a chain DataFrame and returns a list of
``simulation.Leg`` objects ready to feed the backtest engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from .simulation import Leg


@dataclass
class LegSpec:
    """Per-leg target description."""
    right: Literal["C", "P"]
    side: Literal["long", "short"]
    dte: int
    delta_target: float | None = None    # |Δ|; None → ATM (closest to forward)
    qty: int = 1                         # contracts (always positive; sign comes from `side`)


@dataclass
class StructureParams:
    legs: list[LegSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "legs": [
                {"right": l.right, "side": l.side, "dte": int(l.dte),
                 "delta_target": (None if l.delta_target is None else float(l.delta_target)),
                 "qty": int(l.qty)}
                for l in self.legs
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructureParams":
        return cls(
            legs=[LegSpec(
                right=L["right"], side=L["side"], dte=int(L["dte"]),
                delta_target=(None if L.get("delta_target") in (None, "")
                              else float(L["delta_target"])),
                qty=int(L.get("qty", 1)),
            ) for L in d.get("legs", [])],
        )


# ---------------------------------------------------------------------------

def _pick_expiry(chain: pd.DataFrame, target_dte: int) -> pd.Timestamp | None:
    exp = chain.dropna(subset=["tau"])
    exp = exp[exp["tau"] > 0]
    if exp.empty:
        return None
    by_exp = exp.groupby("expiry")["tau"].first()
    target_tau = target_dte / 365.25
    diff = (by_exp - target_tau).abs()
    return diff.idxmin()


def _pick_strike(side_df: pd.DataFrame, target_delta: float | None,
                 forward: float | None) -> pd.Series | None:
    """Pick a single contract row from one side (calls or puts).

    If ``target_delta`` is None: ATM = strike closest to forward.
    Else: contract whose ``|delta|`` is closest to ``target_delta``.
    """
    s = side_df.dropna(subset=["mid", "delta", "strike"]).copy()
    s = s[s["mid"] > 0]
    if s.empty:
        return None
    if target_delta is None:
        F = forward if (forward is not None and np.isfinite(forward)) \
            else float(s["spot"].iloc[0])
        idx = (s["strike"] - F).abs().idxmin()
    else:
        idx = (s["delta"].abs() - target_delta).abs().idxmin()
    return s.loc[idx]


def build_legs(params: StructureParams, chain: pd.DataFrame, day: date) \
        -> list[Leg] | None:
    """Pick concrete contracts on ``day`` matching ``params``. ``chain`` is the
    enriched chain DataFrame for that date.
    """
    if chain is None or chain.empty or not params.legs:
        return None

    unique_dtes = sorted({L.dte for L in params.legs})
    expiry_by_dte: dict[int, pd.Timestamp] = {}
    for d in unique_dtes:
        e = _pick_expiry(chain, d)
        if e is None:
            return None
        expiry_by_dte[d] = e

    legs: list[Leg] = []
    for spec in params.legs:
        expiry = expiry_by_dte[spec.dte]
        sub = chain[chain["expiry"] == expiry]
        side_df = sub[sub["right"] == spec.right]
        if side_df.empty:
            return None
        F = float(side_df["forward"].iloc[0]) if "forward" in side_df.columns and \
            np.isfinite(side_df["forward"].iloc[0]) else None
        row = _pick_strike(side_df, spec.delta_target, F)
        if row is None:
            return None
        mid = float(row["mid"])
        if not np.isfinite(mid) or mid <= 0:
            return None
        sign = +1 if spec.side == "long" else -1
        legs.append(Leg(
            symbol=str(row["symbol"]),
            right=str(row["right"]),
            strike=float(row["strike"]),
            expiry=pd.Timestamp(row["expiry"]),
            qty=sign * abs(int(spec.qty)),
            entry_price=mid,
        ))
    return legs
