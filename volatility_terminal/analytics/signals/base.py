"""Signal ABC + Composite/Smoothed wrappers + Condition/Rule."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


CompOp = Literal["diff", "ratio", "sum", "product"]
SmoothKind = Literal["SMA", "EMA", "MEDIAN", "HAMPEL"]
TransformKind = Literal["RANK", "PERCENTILE", "ZSCORE"]
ThresholdOp = Literal[">", "<", ">=", "<=", "cross_up", "cross_down"]
RuleCombine = Literal["AND", "OR"]


class Signal(ABC):
    """Abstract signal node. Subclasses implement ``_compute`` and ``to_dict``."""

    # Override in subclasses with the registry kind string.
    KIND: str = ""

    @abstractmethod
    def _compute(self, ticker: str) -> pd.Series:
        """Return a Series indexed by tz-naive midnight-normalized DatetimeIndex."""

    def series(self, ticker: str) -> pd.Series:
        from .cache import get_or_compute
        return get_or_compute(self, ticker)

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Signal":
        ...

    @abstractmethod
    def label(self) -> str:
        ...

    def hash_key(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------

class SmoothedSignal(Signal):
    KIND = "smoothed"

    def __init__(self, child: Signal, kind: SmoothKind, window: int,
                 k: float = 3.0):
        self.child = child
        self.kind = kind
        self.window = int(window)
        # Only used by HAMPEL — number of MADs from the rolling median
        # beyond which a point is considered an outlier.
        self.k = float(k)

    def _compute(self, ticker: str) -> pd.Series:
        s = self.child.series(ticker).astype(float)
        if s.empty:
            return s
        min_p = max(1, self.window // 2)
        if self.kind == "SMA":
            return s.rolling(self.window, min_periods=min_p).mean()
        if self.kind == "EMA":
            return s.ewm(span=self.window, adjust=False).mean()
        if self.kind == "MEDIAN":
            return s.rolling(self.window, min_periods=min_p).median()
        if self.kind == "HAMPEL":
            # Rolling median; rolling MAD scaled by 1.4826 so MAD ≈ σ for normal data.
            med = s.rolling(self.window, min_periods=min_p, center=True).median()
            mad = (s - med).abs().rolling(
                self.window, min_periods=min_p, center=True).median() * 1.4826
            # Replace outliers (|x − med| > k·MAD) with the median.
            mask = (s - med).abs() > (self.k * mad)
            cleaned = s.where(~mask.fillna(False), med)
            return cleaned
        raise ValueError(f"Unknown smoother kind: {self.kind}")

    def to_dict(self) -> dict:
        d = {"kind": self.KIND, "smooth": self.kind, "window": self.window,
             "child": self.child.to_dict()}
        if self.kind == "HAMPEL":
            d["k"] = self.k
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SmoothedSignal":
        from .registry import signal_from_dict
        return cls(signal_from_dict(d["child"]), d["smooth"],
                   int(d["window"]), float(d.get("k", 3.0)))

    def label(self) -> str:
        if self.kind == "HAMPEL":
            return f"HAMPEL{self.window}k{self.k:g}({self.child.label()})"
        return f"{self.kind}{self.window}({self.child.label()})"


class TransformedSignal(Signal):
    """Wraps a signal with a stationarizing transform over a rolling lookback.

    - ``RANK``: tastytrade-style rolling rank, ``(x - min) / (max - min) * 100``.
      Range 0–100. Insensitive to mean drift, sensitive to range.
    - ``PERCENTILE``: ``% of values <= current`` in the window. Range 0–100.
      Smoother than rank, but slow to react to new highs.
    - ``ZSCORE``: ``(x - rolling_mean) / rolling_std``. Native units of σ.
    """
    KIND = "transformed"

    def __init__(self, child: Signal, kind: TransformKind, lookback: int):
        self.child = child
        self.kind = kind
        self.lookback = int(lookback)

    def _compute(self, ticker: str) -> pd.Series:
        s = self.child.series(ticker).astype(float)
        if s.empty:
            return s
        n = self.lookback
        min_p = max(20, n // 4)
        if self.kind == "RANK":
            def _rank(w):
                lo, hi = np.nanmin(w), np.nanmax(w)
                if not np.isfinite(hi - lo) or hi - lo == 0:
                    return np.nan
                return (w[-1] - lo) / (hi - lo) * 100.0
            return s.rolling(n, min_periods=min_p).apply(_rank, raw=True)
        if self.kind == "PERCENTILE":
            return s.rolling(n, min_periods=min_p).apply(
                lambda w: float(np.sum(w <= w[-1]) / len(w) * 100.0), raw=True)
        if self.kind == "ZSCORE":
            mean = s.rolling(n, min_periods=min_p).mean()
            std = s.rolling(n, min_periods=min_p).std(ddof=0)
            return (s - mean) / std.replace(0, np.nan)
        raise ValueError(f"Unknown transform kind: {self.kind}")

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "transform": self.kind,
                "lookback": self.lookback, "child": self.child.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "TransformedSignal":
        from .registry import signal_from_dict
        return cls(signal_from_dict(d["child"]), d["transform"],
                   int(d["lookback"]))

    def label(self) -> str:
        prefix = {"RANK": "RANK", "PERCENTILE": "PCTL", "ZSCORE": "Z"}[self.kind]
        return f"{prefix}{self.lookback}({self.child.label()})"


class CompositeSignal(Signal):
    KIND = "composite"

    def __init__(self, left: Signal, right: Signal, op: CompOp):
        self.left = left
        self.right = right
        self.op = op

    def _compute(self, ticker: str) -> pd.Series:
        a = self.left.series(ticker).astype(float)
        b = self.right.series(ticker).astype(float)
        if a.empty or b.empty:
            return pd.Series(dtype=float)
        # pandas aligns by index automatically
        if self.op == "diff":
            return a - b
        if self.op == "sum":
            return a + b
        if self.op == "product":
            return a * b
        # ratio
        out = a / b.replace(0, np.nan)
        return out

    def to_dict(self) -> dict:
        return {"kind": self.KIND, "op": self.op,
                "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeSignal":
        from .registry import signal_from_dict
        return cls(
            signal_from_dict(d["left"]),
            signal_from_dict(d["right"]),
            d["op"],
        )

    def label(self) -> str:
        sym = {"diff": "-", "ratio": "/", "sum": "+", "product": "*"}[self.op]
        return f"({self.left.label()} {sym} {self.right.label()})"


# ---------------------------------------------------------------------------
# Persistence-safe condition / rule configs (stored, not evaluated directly)

@dataclass
class ConditionConfig:
    signal: dict                 # signal.to_dict()
    op: ThresholdOp
    threshold: float

    def to_dict(self) -> dict:
        return {"signal": self.signal, "op": self.op, "threshold": float(self.threshold)}

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionConfig":
        return cls(signal=d["signal"], op=d["op"], threshold=float(d["threshold"]))


@dataclass
class RuleConfig:
    conditions: list[ConditionConfig] = field(default_factory=list)
    combine: RuleCombine = "AND"

    def to_dict(self) -> dict:
        return {"combine": self.combine,
                "conditions": [c.to_dict() for c in self.conditions]}

    @classmethod
    def from_dict(cls, d: dict) -> "RuleConfig":
        return cls(
            conditions=[ConditionConfig.from_dict(c) for c in d.get("conditions", [])],
            combine=d.get("combine", "AND"),
        )

    def is_empty(self) -> bool:
        return not self.conditions


# ---------------------------------------------------------------------------
# Active condition / rule (built from a config + materialized signals)

@dataclass
class Condition:
    signal: Signal
    op: ThresholdOp
    threshold: float

    def evaluate(self, ticker: str) -> pd.Series:
        s = self.signal.series(ticker).astype(float)
        if s.empty:
            return pd.Series(dtype=bool)
        t = self.threshold
        if self.op == ">":
            return s > t
        if self.op == "<":
            return s < t
        if self.op == ">=":
            return s >= t
        if self.op == "<=":
            return s <= t
        prev = s.shift(1)
        if self.op == "cross_up":
            return (prev <= t) & (s > t)
        if self.op == "cross_down":
            return (prev >= t) & (s < t)
        raise ValueError(f"Unknown op: {self.op}")


@dataclass
class Rule:
    conditions: list[Condition]
    combine: RuleCombine = "AND"

    def evaluate(self, ticker: str) -> pd.Series:
        if not self.conditions:
            return pd.Series(dtype=bool)
        masks = [c.evaluate(ticker) for c in self.conditions]
        # Align all on the same index; missing → False
        idx = masks[0].index
        for m in masks[1:]:
            idx = idx.union(m.index)
        aligned = [m.reindex(idx).fillna(False).astype(bool) for m in masks]
        if self.combine == "OR":
            out = aligned[0]
            for m in aligned[1:]:
                out = out | m
        else:  # AND
            out = aligned[0]
            for m in aligned[1:]:
                out = out & m
        return out


def rule_from_config(cfg: RuleConfig) -> Rule:
    from .registry import signal_from_dict
    conds = [
        Condition(signal=signal_from_dict(c.signal), op=c.op, threshold=c.threshold)
        for c in cfg.conditions
    ]
    return Rule(conditions=conds, combine=cfg.combine)
