"""SYNTHETIC TEST FIXTURES ONLY. Never imported by app code, never written to the live database,
never shown as live results."""
from __future__ import annotations
import numpy as np
import pandas as pd

SYNTHETIC = True


def make_synthetic_candles(n: int = 3000, seed: int = 0, start_ts: int = 1_700_000_040 - 1_700_000_040 % 60,
                           autocorr: float = 0.0, sigma: float = 1e-4, gaps_at: tuple[int, ...] = ()
                           ) -> pd.DataFrame:
    """Candles whose 1-minute returns follow AR(1) with coefficient `autocorr` (0 = random walk)."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        r[i] = autocorr * r[i - 1] + eps[i]
    close = 1.10 * np.exp(np.cumsum(r))
    open_ = np.r_[1.10, close[:-1]]
    hi = np.maximum(open_, close) + np.abs(rng.normal(0, sigma * 0.3, n))
    lo = np.minimum(open_, close) - np.abs(rng.normal(0, sigma * 0.3, n))
    ts = start_ts + np.arange(n) * 60
    df = pd.DataFrame({"ts": ts, "open": open_, "high": hi, "low": lo, "close": close,
                       "volume": rng.uniform(5, 20, n)})
    if gaps_at:
        df = df.drop(index=list(gaps_at)).reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df
