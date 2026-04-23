# Volatility Terminal

Interactive desktop terminal for historical equity option volatility analysis —
term structure, skew, IV surface, and variance risk premium — powered by the
Alpaca historical options API (free tier).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python -m volatility_terminal
```

On first launch you will be prompted for your Alpaca API key and secret. They
are persisted via `QSettings` so you only enter them once.

## Layout

```
volatility_terminal/
    data/        Alpaca + FRED clients, parquet cache, chain fetcher
    pricing/     Black-Scholes, OCC parser, rates, parity, surface grid
    analytics/   Term, skew, realized vol, VRP, IV timeseries
    ui/          PyQt5 + pyqtgraph main window and tabs
cache/           Local parquet cache (created at runtime, gitignored)
```

## Data

- Source: Alpaca historical options, end-of-day daily bars per OPRA contract.
- Coverage: from Alpaca OPRA start (~Feb 2024) through yesterday.
- Risk-free rate: FRED daily Treasury yields, interpolated per expiry tenor.
- Dividend yield: inferred per snapshot from put-call parity at the ATM strike.
