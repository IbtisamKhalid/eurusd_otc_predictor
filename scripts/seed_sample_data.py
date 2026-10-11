"""Seed the local database with realistic synthetic completed EURUSD-OTC candles.

This enables full local offline evaluation, model training, walk-forward backtesting,
and interactive dashboard exploration without needing an active OTCharts API subscription.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import config
from api.schemas import Candle
from config import DB_PATH, SYMBOL, TIMEFRAME_S, VENUE
from data.database import Database


def generate_candles(n: int = 3500, seed: int = 42, base_price: float = 1.0850) -> list[Candle]:
    """Generate n 1-minute candles ending at the last completed minute."""
    now = int(time.time())
    # round down to nearest completed minute
    last_completed = (now // 60) * 60 - 60
    start_ts = last_completed - (n - 1) * 60

    rng = np.random.default_rng(seed)
    # realistic OTC volatility: sigma ~ 0.0001 (1 pip per min) with slight mean reversion
    sigma = 0.00012
    returns = rng.normal(0, sigma, n)
    # add slight autocorrelation for realistic micro-regimes
    for i in range(1, n):
        returns[i] += 0.03 * returns[i - 1]

    log_prices = np.log(base_price) + np.cumsum(returns)
    closes = np.exp(log_prices)
    opens = np.r_[base_price, closes[:-1]]

    candles: list[Candle] = []
    for i in range(n):
        ts = int(start_ts + i * 60)
        o = float(opens[i])
        c = float(closes[i])
        wick_high = abs(float(rng.normal(0, sigma * 0.4)))
        wick_low = abs(float(rng.normal(0, sigma * 0.4)))
        h = max(o, c) + wick_high
        l = min(o, c) - wick_low
        v = float(rng.uniform(10, 150))
        candles.append(Candle(ts=ts, open=round(o, 5), high=round(h, 5),
                              low=round(l, 5), close=round(c, 5), volume=round(v, 1)))

    return candles


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed database with synthetic EURUSD-OTC candles.")
    parser.add_argument("--bars", type=int, default=3500, help="Number of 1m bars to generate (min 2000 for training).")
    args = parser.parse_args()

    db = Database(DB_PATH)
    print(f"Generating {args.bars} synthetic candles ending near current time...")
    candles = generate_candles(n=args.bars)
    inserted, conflicts = db.insert_candles(VENUE, SYMBOL, TIMEFRAME_S, candles)
    print(f"Inserted {inserted} candles into {DB_PATH} for {VENUE}:{SYMBOL}:{TIMEFRAME_S}s.")

    total = len(db.load_candles(VENUE, SYMBOL, TIMEFRAME_S))
    print(f"Total stored candles in DB: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
