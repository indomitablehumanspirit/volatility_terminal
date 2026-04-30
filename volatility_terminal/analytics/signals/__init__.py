"""Composable signal expression layer for the Backtest tab.

A ``Signal`` is any object exposing ``series(ticker) -> pd.Series`` (DatetimeIndex,
tz-naive, midnight-normalized). Signals compose into ``CompositeSignal`` (binary
op of two signals) and ``SmoothedSignal`` (SMA/EMA wrapper). ``Condition``s test
a signal against a threshold; ``Rule``s AND/OR conditions.
"""
from .base import (
    Signal, SmoothedSignal, CompositeSignal, TransformedSignal,
    Condition, Rule,
    ConditionConfig, RuleConfig,
)
from .primitives import (
    AtmIvSignal, RealizedVolSignal, IvRvRatioSignal, VrpSignal,
    RiskReversalSignal, ButterflySignal, ForwardVolSignal,
)
from .registry import (
    SIGNAL_TYPES, signal_from_dict, predefined_signals, register_signal,
)
from .cache import clear_signal_cache

__all__ = [
    "Signal", "SmoothedSignal", "CompositeSignal", "TransformedSignal",
    "Condition", "Rule", "ConditionConfig", "RuleConfig",
    "AtmIvSignal", "RealizedVolSignal", "IvRvRatioSignal", "VrpSignal",
    "RiskReversalSignal", "ButterflySignal", "ForwardVolSignal",
    "SIGNAL_TYPES", "signal_from_dict", "predefined_signals", "register_signal",
    "clear_signal_cache",
]
