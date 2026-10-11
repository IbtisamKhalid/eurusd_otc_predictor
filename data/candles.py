"""Candle helpers: completeness, DataFrame conversion, outcome classification."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from api.schemas import Candle

COLS = ["ts", "open", "high", "low", "close", "volume"]


def is_complete(c: Candle, now_server: float, tf: int = 60) -> bool:
    """A bar is complete only once its full interval has elapsed (conservative whether the
    provider's `time` marks bar open or close; never leaks a forming bar)."""
    return now_server >= c.ts + tf


def split_complete(candles: Sequence[Candle], now_server: float, tf: int = 60
                   ) -> tuple[list[Candle], list[Candle]]:
    done = [c for c in candles if is_complete(c, now_server, tf)]
    forming = [c for c in candles if not is_complete(c, now_server, tf)]
    return done, forming


def to_frame(candles: Sequence[Candle]) -> pd.DataFrame:
    df = pd.DataFrame([{"ts": c.ts, "open": c.open, "high": c.high, "low": c.low,
                        "close": c.close, "volume": np.nan if c.volume is None else c.volume}
                       for c in candles], columns=COLS)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    df["ts"] = df["ts"].astype("int64")
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def classify(open_: float, close: float, tol: float) -> str:
    d = close - open_
    if abs(d) <= tol:
        return "DOJI"
    return "UP" if d > 0 else "DOWN"


def label_next(df: pd.DataFrame, tol: float, tf: int = 60) -> pd.Series:
    """Label for row t = outcome of bar t+1 (UP/DOWN/DOJI). NaN when t+1 is not the
    immediately following minute (gap) or does not exist yet."""
    nxt_open, nxt_close = df["open"].shift(-1), df["close"].shift(-1)
    contiguous = (df["ts"].shift(-1) - df["ts"]) == tf
    d = nxt_close - nxt_open
    lab = pd.Series(np.where(d.abs() <= tol, "DOJI", np.where(d > 0, "UP", "DOWN")),
                    index=df.index, dtype="object")
    lab[~contiguous | nxt_open.isna()] = np.nan
    return lab
