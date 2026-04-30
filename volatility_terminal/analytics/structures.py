"""Generic multi-leg structure builder.

Given a ``StructureParams`` describing the structure kind, direction, qty, and
per-leg DTE/delta targets, ``build_legs`` picks concrete contracts off a chain
DataFrame and returns a list of ``simulation.Leg`` objects ready to feed the
backtest engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from .simulation import Leg


StructureKind = Literal[
    "naked_call", "naked_put",
    "straddle", "strangle",
    "vertical_call", "vertical_put",
    "calendar_call", "calendar_put",
    "butterfly_call", "butterfly_put",
    "iron_condor",
]

Direction = Literal["long", "short"]


@dataclass
class LegSpec:
    """Per-leg target description.

    ``right`` and ``side`` are typically determined by the structure kind, but
    are stored explicitly so ``build_legs`` is uniform across structures.
    ``delta_target`` is the absolute delta (positive); ``side`` indicates long
    (+) or short (-) and combines with the structure's overall direction.
    """
    right: Literal["C", "P"]
    side: Literal["long", "short"]      # leg direction relative to structure
    dte: int                             # target days to expiry
    delta_target: float | None = None    # |Δ|; None → ATM (closest to forward)


@dataclass
class StructureParams:
    kind: StructureKind
    direction: Direction = "short"       # overall long/short structure
    qty: int = 1                         # contracts per leg
    legs: list[LegSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "direction": self.direction, "qty": int(self.qty),
            "legs": [
                {"right": l.right, "side": l.side, "dte": int(l.dte),
                 "delta_target": (None if l.delta_target is None else float(l.delta_target))}
                for l in self.legs
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructureParams":
        return cls(
            kind=d["kind"],
            direction=d.get("direction", "short"),
            qty=int(d.get("qty", 1)),
            legs=[LegSpec(
                right=L["right"], side=L["side"], dte=int(L["dte"]),
                delta_target=(None if L.get("delta_target") in (None, "")
                              else float(L["delta_target"])),
            ) for L in d.get("legs", [])],
        )


# ---------------------------------------------------------------------------
# Default leg templates per structure kind. The UI calls
# ``default_legs(kind)`` to populate the leg-spec rows when the user picks a
# structure, then mutates DTE / delta as desired.

def default_legs(kind: StructureKind, dte: int = 30) -> list[LegSpec]:
    if kind == "naked_call":
        return [LegSpec("C", "short", dte, None)]   # ATM by default
    if kind == "naked_put":
        return [LegSpec("P", "short", dte, None)]
    if kind == "straddle":
        return [LegSpec("C", "short", dte, None),
                LegSpec("P", "short", dte, None)]
    if kind == "strangle":
        return [LegSpec("C", "short", dte, 0.25),
                LegSpec("P", "short", dte, 0.25)]
    if kind == "vertical_call":
        # short the closer leg, long the farther wing (credit call spread)
        return [LegSpec("C", "short", dte, 0.30),
                LegSpec("C", "long",  dte, 0.15)]
    if kind == "vertical_put":
        return [LegSpec("P", "short", dte, 0.30),
                LegSpec("P", "long",  dte, 0.15)]
    if kind == "calendar_call":
        return [LegSpec("C", "short", dte,        None),   # front ATM
                LegSpec("C", "long",  dte * 2,    None)]   # back ATM
    if kind == "calendar_put":
        return [LegSpec("P", "short", dte,        None),
                LegSpec("P", "long",  dte * 2,    None)]
    if kind == "butterfly_call":
        return [LegSpec("C", "long",  dte, 0.40),
                LegSpec("C", "short", dte, 0.50),    # body (qty 2 baked in below)
                LegSpec("C", "short", dte, 0.50),
                LegSpec("C", "long",  dte, 0.15)]
    if kind == "butterfly_put":
        return [LegSpec("P", "long",  dte, 0.40),
                LegSpec("P", "short", dte, 0.50),
                LegSpec("P", "short", dte, 0.50),
                LegSpec("P", "long",  dte, 0.15)]
    if kind == "iron_condor":
        return [LegSpec("C", "short", dte, 0.25),
                LegSpec("C", "long",  dte, 0.10),
                LegSpec("P", "short", dte, 0.25),
                LegSpec("P", "long",  dte, 0.10)]
    raise ValueError(f"Unknown structure kind: {kind}")


def leg_label(kind: StructureKind, idx: int) -> str:
    """Human-readable label for the n-th leg of a structure (used in the UI)."""
    legs = default_legs(kind)
    if idx >= len(legs):
        return f"Leg {idx+1}"
    L = legs[idx]
    return f"{L.side.title()} {L.right} (DTE {L.dte})"


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

    # For each unique target DTE referenced by the spec, resolve to a concrete expiry
    unique_dtes = sorted({L.dte for L in params.legs})
    expiry_by_dte: dict[int, pd.Timestamp] = {}
    for d in unique_dtes:
        e = _pick_expiry(chain, d)
        if e is None:
            return None
        expiry_by_dte[d] = e

    # Direction sign helper: structure direction × leg side.
    # "short" structure flips long↔short.
    def signed_qty(leg_side: str) -> int:
        sign = +1 if leg_side == "long" else -1
        if params.direction == "short":
            sign = -sign
        return sign * abs(params.qty)

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
        legs.append(Leg(
            symbol=str(row["symbol"]),
            right=str(row["right"]),
            strike=float(row["strike"]),
            expiry=pd.Timestamp(row["expiry"]),
            qty=signed_qty(spec.side),
            entry_price=mid,
        ))
    return legs
