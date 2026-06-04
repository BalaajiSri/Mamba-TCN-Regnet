#!/usr/bin/env python3
"""Download daily OHLCV data for all paper assets.

Usage:
    python scripts/fetch_asset_data.py            # downloads ETH + BNB
    python scripts/fetch_asset_data.py --all      # downloads ETH + BNB + SOL + XRP

Output: data/<symbol>_daily_full.csv  (same format as btc_daily_full.csv)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ASSETS = {
    "eth": ("ETH-USD", "2018-09-17", "2026-03-14"),
    "bnb": ("BNB-USD", "2018-09-17", "2026-03-14"),
    "sol": ("SOL-USD", "2020-04-11", "2026-03-14"),
    "xrp": ("XRP-USD", "2018-09-17", "2026-03-14"),
}

DEFAULT_ASSETS = ["eth", "bnb"]

def download(name: str, ticker: str, start: str, end: str, out_dir: Path) -> None:
    out_path = out_dir / f"{name}_daily_full.csv"
    if out_path.exists():
        print(f"  {out_path} already exists, skipping.")
        return
    print(f"  Downloading {ticker} {start} -> {end} ...", end=" ", flush=True)
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        print("FAILED (empty dataframe)")
        return
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df.to_csv(out_path)
    print(f"{len(df)} rows saved to {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download all assets incl. SOL, XRP")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data"
    out_dir.mkdir(exist_ok=True)

    assets = list(ASSETS.keys()) if args.all else DEFAULT_ASSETS
    print(f"Fetching {assets} ...")
    for name in assets:
        ticker, start, end = ASSETS[name]
        download(name, ticker, start, end, out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
