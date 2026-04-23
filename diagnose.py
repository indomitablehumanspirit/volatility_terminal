"""Run chain_fetcher end-to-end and print row counts at every stage."""
import sys
from datetime import date, datetime, time, timezone

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication

from volatility_terminal.ui.creds_dialog import load_creds
from volatility_terminal.data.alpaca_client import AlpacaCreds, AlpacaOptionsData
from volatility_terminal.data import cache
from volatility_terminal.pricing.rates import RateCurve
from volatility_terminal.pricing import bs, parity
from volatility_terminal.pricing.occ import parse_occ


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    day = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 3, 13)
    _ = QApplication(sys.argv)
    creds = load_creds()
    if not creds:
        print("No creds. Run the app once.")
        return

    alpaca = AlpacaOptionsData(AlpacaCreds(*creds))
    rates = RateCurve(cache_path=cache.rates_path())

    print(f"[1] list_contracts {ticker} {day}")
    contracts = alpaca.list_contracts(ticker, as_of=day, expiration_after=day)
    print(f"    contracts rows: {len(contracts)}")
    if contracts.empty:
        return

    # Limit to nearest 6 expiries for a fast diagnostic
    exp_keep = sorted(contracts["expiration"].unique())[:6]
    contracts = contracts[contracts["expiration"].isin(exp_keep)].copy()
    print(f"    narrowed to {len(contracts)} contracts across {len(exp_keep)} expiries "
          f"({[pd.Timestamp(e).date() for e in exp_keep]})")

    print(f"\n[2] get_daily_bars (batched)")
    bars = alpaca.get_daily_bars(contracts["symbol"].tolist(), day)
    print(f"    bar rows: {len(bars)}")
    if bars.empty:
        print("    !! no bars — all contracts had zero volume? Try a more liquid date.")
        return
    print(f"    sample:\n{bars.head(3).to_string(index=False)}")

    print(f"\n[3] spot price")
    under = alpaca.get_daily_stock_bars(ticker, day, day)
    print(f"    stock rows: {len(under)}")
    if under.empty:
        print("    !! no stock bar on this day")
        return
    spot = float(under['close'].iloc[-1])
    print(f"    spot close: {spot}")

    print(f"\n[4] merge contracts + bars")
    df = contracts.merge(bars[["symbol", "close", "volume"]], on="symbol", how="inner")
    df = df[df["close"] > 0].copy()
    print(f"    after inner merge & close>0: {len(df)} rows")

    df["mid"] = df["close"]
    parsed = df["symbol"].map(parse_occ)
    df["expiry_ts"] = [p[0] for p in parsed]
    n_unparsed = df["expiry_ts"].isna().sum()
    print(f"    OCC parse failures: {n_unparsed}")
    exp_from_field = pd.to_datetime(df["expiration"]).dt.tz_localize("UTC") + pd.Timedelta(hours=20)
    df["expiry_ts"] = df["expiry_ts"].fillna(exp_from_field)
    df["expiry"] = df["expiry_ts"].dt.normalize()

    as_of_ts = pd.Timestamp(datetime.combine(day, time(20, 0), tzinfo=timezone.utc))
    df["tau"] = (df["expiry_ts"] - as_of_ts).dt.total_seconds() / (365.25 * 86400.0)
    print(f"    tau range: {df['tau'].min():.4f} .. {df['tau'].max():.4f}")
    df = df[df["tau"] > 0].copy()
    print(f"    after tau>0: {len(df)}")
    df["spot"] = spot

    print(f"\n[5] parity: infer forward & q per expiry")
    r_of_tau = lambda tau: rates.r_at(day, tau)
    print(f"    r(30d)={r_of_tau(30/365.25):.4f}  r(180d)={r_of_tau(180/365.25):.4f}")
    per_exp = parity.infer_forward_and_q(df, spot, r_of_tau)
    print(f"    expiries with forward/q: {len(per_exp)}")
    print(per_exp.head().to_string())
    if per_exp.empty:
        print("    !! parity returned nothing — insufficient call/put pairs.")
        return

    df = df.merge(per_exp[["r", "forward", "q"]], left_on="expiry_ts",
                  right_index=True, how="left")
    print(f"    after merge with per_exp: {len(df)}")
    df = df.dropna(subset=["r", "forward", "q"])
    print(f"    after dropna r/forward/q: {len(df)}")

    print(f"\n[6] implied vol for each row")
    ivs = []
    for row in df.head(10).itertuples(index=False):
        iv = bs.implied_vol(row.mid, row.spot, row.strike, row.tau,
                            row.r, row.right, row.q)
        ivs.append((row.symbol, row.strike, row.right, row.mid, iv))
    for s, k, r, m, iv in ivs:
        print(f"    {s}  K={k}  {r}  mid={m}  iv={iv}")


if __name__ == "__main__":
    main()
