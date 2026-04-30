"""User-facing library of named signals: predefined + persisted custom.

Custom entries are stored in QSettings as a JSON list under
``backtest/custom_signals``. Predefined entries are virtual; they are always
present and not editable.

Each entry: ``{"name": str, "signal": signal.to_dict()}``. The display name is
distinct from ``signal.label()`` because multiple library entries can share a
labelling formula but have different aliases.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from PyQt5.QtCore import QSettings

from ..analytics.signals import (
    AtmIvSignal, ButterflySignal, ForwardVolSignal, IvRvRatioSignal,
    RealizedVolSignal, RiskReversalSignal, Signal, VrpSignal,
    signal_from_dict,
)


SETTINGS_ORG = "VolatilityTerminal"
SETTINGS_APP = "VolatilityTerminal"
SETTINGS_KEY = "backtest/custom_signals"


@dataclass
class LibraryEntry:
    name: str
    signal: Signal
    builtin: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "signal": self.signal.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "LibraryEntry":
        return cls(name=d["name"], signal=signal_from_dict(d["signal"]),
                   builtin=False)


def _builtin_entries() -> list[LibraryEntry]:
    return [
        LibraryEntry("ATM IV (30d)",            AtmIvSignal(30),               True),
        LibraryEntry("ATM IV (60d)",            AtmIvSignal(60),               True),
        LibraryEntry("ATM IV (90d)",            AtmIvSignal(90),               True),
        LibraryEntry("Realized Vol (20d)",      RealizedVolSignal(20),         True),
        LibraryEntry("Realized Vol (30d)",      RealizedVolSignal(30),         True),
        LibraryEntry("IV/RV ratio (30/30)",     IvRvRatioSignal(30, 30),       True),
        LibraryEntry("VRP (30/30)",             VrpSignal(30, 30),             True),
        LibraryEntry("Risk reversal 25Δ (30d)", RiskReversalSignal(30, 0.25),  True),
        LibraryEntry("Butterfly 25Δ (30d)",     ButterflySignal(30, 0.25),     True),
        LibraryEntry("Forward vol 30/60",       ForwardVolSignal(30, 60),      True),
        LibraryEntry("Forward vol 60/90",       ForwardVolSignal(60, 90),      True),
    ]


class SignalLibrary:
    def __init__(self):
        self._builtin = _builtin_entries()
        self._custom: list[LibraryEntry] = []
        self._load_custom()

    # -- persistence --
    def _settings(self) -> QSettings:
        return QSettings(SETTINGS_ORG, SETTINGS_APP)

    def _load_custom(self) -> None:
        s = self._settings()
        raw = s.value(SETTINGS_KEY, "", type=str)
        if not raw:
            self._custom = []
            return
        try:
            data = json.loads(raw)
            self._custom = [LibraryEntry.from_dict(d) for d in data]
        except Exception:
            self._custom = []

    def _save_custom(self) -> None:
        s = self._settings()
        data = [e.to_dict() for e in self._custom]
        s.setValue(SETTINGS_KEY, json.dumps(data))

    # -- query --
    def entries(self) -> list[LibraryEntry]:
        return list(self._builtin) + list(self._custom)

    def names(self) -> list[str]:
        return [e.name for e in self.entries()]

    def get(self, name: str) -> LibraryEntry | None:
        for e in self.entries():
            if e.name == name:
                return e
        return None

    # -- mutations (only on custom) --
    def add(self, entry: LibraryEntry) -> None:
        # Replace any existing custom of the same name; built-in names protected.
        if any(e.name == entry.name for e in self._builtin):
            raise ValueError(f"Cannot use built-in name: {entry.name}")
        self._custom = [e for e in self._custom if e.name != entry.name]
        self._custom.append(LibraryEntry(name=entry.name, signal=entry.signal,
                                         builtin=False))
        self._save_custom()

    def remove(self, name: str) -> bool:
        before = len(self._custom)
        self._custom = [e for e in self._custom if e.name != name]
        changed = len(self._custom) != before
        if changed:
            self._save_custom()
        return changed
