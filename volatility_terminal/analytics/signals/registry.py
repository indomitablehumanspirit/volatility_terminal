"""Signal type registry. Adding a new signal type = one entry here."""
from __future__ import annotations

from .base import CompositeSignal, Signal, SmoothedSignal, TransformedSignal
from .primitives import (
    AtmIvSignal, ButterflySignal, ForwardVolSignal, IvRvRatioSignal,
    RealizedVolSignal, RiskReversalSignal, VrpSignal,
)


SIGNAL_TYPES: dict[str, type[Signal]] = {
    AtmIvSignal.KIND: AtmIvSignal,
    RealizedVolSignal.KIND: RealizedVolSignal,
    IvRvRatioSignal.KIND: IvRvRatioSignal,
    VrpSignal.KIND: VrpSignal,
    RiskReversalSignal.KIND: RiskReversalSignal,
    ButterflySignal.KIND: ButterflySignal,
    ForwardVolSignal.KIND: ForwardVolSignal,
    SmoothedSignal.KIND: SmoothedSignal,
    CompositeSignal.KIND: CompositeSignal,
    TransformedSignal.KIND: TransformedSignal,
}


def register_signal(cls: type[Signal]) -> None:
    SIGNAL_TYPES[cls.KIND] = cls


def signal_from_dict(d: dict) -> Signal:
    kind = d.get("kind")
    if kind not in SIGNAL_TYPES:
        raise ValueError(f"Unknown signal kind: {kind!r}")
    return SIGNAL_TYPES[kind].from_dict(d)


def predefined_signals() -> list[Signal]:
    """Default signals offered in the UI dropdown on first launch."""
    return [
        AtmIvSignal(dte=30),
        AtmIvSignal(dte=60),
        RealizedVolSignal(window=30),
        IvRvRatioSignal(dte=30, window=30),
        VrpSignal(dte=30, window=30),
        RiskReversalSignal(dte=30, delta=0.25),
        ButterflySignal(dte=30, delta=0.25),
        ForwardVolSignal(dte1=30, dte2=60),
    ]
