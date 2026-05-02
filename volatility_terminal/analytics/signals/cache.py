"""Per-process memoization for signal series, keyed on (signal_hash, ticker)."""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

_MAX_ENTRIES = 256
_CACHE: OrderedDict[tuple[str, str], pd.Series] = OrderedDict()


def get_or_compute(signal, ticker: str) -> pd.Series:
    key = (signal.hash_key(), ticker.upper())
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    s = signal._compute(ticker)
    if s is not None:
        try:
            s = s.copy()
            if isinstance(s.index, pd.DatetimeIndex):
                if s.index.tz is not None:
                    s.index = s.index.tz_convert(None)
                s.index = s.index.normalize()
        except Exception:
            pass
    _CACHE[key] = s
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)
    return s


def clear_signal_cache() -> None:
    _CACHE.clear()
