"""Fetch real historical candles from Binance public market data into SQLite.

Usage:
    python scripts/collect_binance.py --bars 5000 --symbol EURUSDT
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.binance_client import BinanceClient
from config import DB_PATH, Settings
from data.collector import Collector
from data.database import Database


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect real 1-minute candles from Binance.")
    parser.add_argument("--symbol", default="EURUSDT", help="Binance symbol (e.g. EURUSDT, BTCUSDT).")
    parser.add_argument("--bars", type=int, default=5000, help="Number of historical 1m bars to fetch.")
    args = parser.parse_args()

    s = Settings(venue="binance", symbol=args.symbol, timeframe_s=60)
    db = Database(DB_PATH)
    col = Collector(BinanceClient(), db, s)

    print(f"Fetching up to {args.bars} real 1m candles for {s.symbol} from Binance (venue={s.venue})...")
    res = col.backfill(target_bars=args.bars, page=1000)
    if not res["ok"]:
        print(f"FAILED: {res.get('error')}")
        return 1

    total = len(db.load_candles(s.venue, s.symbol, s.timeframe_s))
    print(f"SUCCESS: Collected and stored {res['inserted']} candles across {res['pages']} pages.")
    print(f"Total stored candles in database for {s.venue}:{s.symbol} = {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
