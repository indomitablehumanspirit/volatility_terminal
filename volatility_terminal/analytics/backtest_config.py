"""Configuration dataclasses for the generic backtest engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .signals.base import RuleConfig
from .simulation import HedgeConfig
from .structures import StructureParams


@dataclass
class BacktestConfig:
    start: date | None = None
    end: date | None = None

    structure: StructureParams = field(
        default_factory=lambda: StructureParams(kind="straddle"))

    entry_rule: RuleConfig = field(default_factory=RuleConfig)
    exit_rule: RuleConfig = field(default_factory=RuleConfig)

    # Structural exits (any single one fires close)
    use_dte_exit: bool = False
    dte_exit_threshold: int = 21
    use_profit_target: bool = False
    profit_target_pct: float = 50.0
    use_stop_loss: bool = False
    stop_loss_pct: float = 200.0

    # Re-arm semantics:
    #  - "any_bar": as soon as flat, attempt entry whenever entry_rule passes
    #  - "edge_only": entry_rule must transition False→True (cross_up-style)
    rearm: str = "any_bar"

    hedge: HedgeConfig = field(default_factory=HedgeConfig)

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "structure": self.structure.to_dict(),
            "entry_rule": self.entry_rule.to_dict(),
            "exit_rule": self.exit_rule.to_dict(),
            "use_dte_exit": self.use_dte_exit,
            "dte_exit_threshold": int(self.dte_exit_threshold),
            "use_profit_target": self.use_profit_target,
            "profit_target_pct": float(self.profit_target_pct),
            "use_stop_loss": self.use_stop_loss,
            "stop_loss_pct": float(self.stop_loss_pct),
            "rearm": self.rearm,
            "hedge": {
                "use_interval": self.hedge.use_interval,
                "interval_days": int(self.hedge.interval_days),
                "use_delta_threshold": self.hedge.use_delta_threshold,
                "delta_threshold": float(self.hedge.delta_threshold),
                "use_spot_move": self.hedge.use_spot_move,
                "spot_move_pct": float(self.hedge.spot_move_pct),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestConfig":
        h = d.get("hedge", {}) or {}
        return cls(
            start=date.fromisoformat(d["start"]) if d.get("start") else None,
            end=date.fromisoformat(d["end"]) if d.get("end") else None,
            structure=StructureParams.from_dict(d.get("structure", {"kind": "straddle"})),
            entry_rule=RuleConfig.from_dict(d.get("entry_rule", {})),
            exit_rule=RuleConfig.from_dict(d.get("exit_rule", {})),
            use_dte_exit=bool(d.get("use_dte_exit", False)),
            dte_exit_threshold=int(d.get("dte_exit_threshold", 21)),
            use_profit_target=bool(d.get("use_profit_target", False)),
            profit_target_pct=float(d.get("profit_target_pct", 50.0)),
            use_stop_loss=bool(d.get("use_stop_loss", False)),
            stop_loss_pct=float(d.get("stop_loss_pct", 200.0)),
            rearm=d.get("rearm", "any_bar"),
            hedge=HedgeConfig(
                use_interval=bool(h.get("use_interval", False)),
                interval_days=int(h.get("interval_days", 5)),
                use_delta_threshold=bool(h.get("use_delta_threshold", False)),
                delta_threshold=float(h.get("delta_threshold", 100.0)),
                use_spot_move=bool(h.get("use_spot_move", False)),
                spot_move_pct=float(h.get("spot_move_pct", 2.0)),
            ),
        )
