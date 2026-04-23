"""OCC21 option-symbol parser."""
from __future__ import annotations

import re
import pandas as pd

OCC_RE = re.compile(
    r"^(?P<root>[A-Z]+)\s*(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<right>[CP])(?P<strike>\d{8})$"
)


def parse_occ(sym: str):
    """Parse an OCC symbol; return (expiry_ts_utc, right, strike) or (None, None, None).

    Expiry is set to 20:00 UTC (~4pm ET standard close) on the expiration date.
    """
    if sym is None:
        return (None, None, None)
    m = OCC_RE.match(sym)
    if not m:
        return (None, None, None)
    expiry = pd.Timestamp(
        f"20{m['yy']}-{m['mm']}-{m['dd']}", tz="UTC"
    ) + pd.Timedelta(hours=20)
    return expiry, m["right"], int(m["strike"]) / 1000.0
