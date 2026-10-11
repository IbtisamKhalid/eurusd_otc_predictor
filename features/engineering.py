"""Leakage-safe feature construction.

Row t describes the market as known at the CLOSE of completed bar t. The label of row t is
the outcome of bar t+1 (see data.candles.label_next). No feature may read bar t+1 or later.
`tests/test_features.py` enforces this by checking that features for rows <= k are identical
whether computed on df[:k+1] or on the full frame.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from features import indicators as ind

RETURN_LAGS = (1, 2, 3, 5, 10)
LOOKBACK = 40            # rows needed before features are all defined (MACD slow+signal etc.)
CONTIGUOUS_WINDOW = 30   # a gap within this many bars invalidates the row


def build_features(df: pd.DataFrame, tf: int = 60, doji_tol: float = 1e-5) -> pd.DataFrame:
    """Return a feature frame indexed like df (includes 'ts'). Rows with undefined features or
    a recent gap have `valid == False`."""
    d = df.reset_index(drop=True)
    c, o, h, l = d["close"], d["open"], d["high"], d["low"]
    f = pd.DataFrame({"ts": d["ts"]})
    logc = np.log(c)
    for k in RETURN_LAGS:
        f[f"ret_{k}"] = logc.diff(k)
    diff = c - o
    bull, bear = diff > doji_tol, diff < -doji_tol
    f["bull_run"], f["bear_run"] = ind.run_length(bull), ind.run_length(bear)
    rng = (h - l).replace(0, np.nan)
    f["body_ratio"] = diff.abs() / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["body_dir"] = np.sign(diff)
    r1 = logc.diff()
    f["vol_10"], f["vol_30"] = r1.rolling(10).std(), r1.rolling(30).std()
    for span in (5, 20):
        f[f"ema_dist_{span}"] = (c - ind.ema(c, span)) / c
    f["ema_cross"] = (ind.ema(c, 5) - ind.ema(c, 20)) / c
    f["rsi_14"] = ind.rsi(c, 14)
    m = ind.macd(c)
    f["macd_hist_n"], f["macd_n"] = m["macd_hist"] / c, m["macd"] / c
    bb = ind.bollinger(c)
    f["bb_pos"], f["bb_width"] = bb["bb_pos"], bb["bb_width"]
    f["atr_n"] = ind.atr(d, 14) / c
    for w in (10, 30):
        f[f"dist_high_{w}"] = (h.rolling(w).max() - c) / c
        f[f"dist_low_{w}"] = (c - l.rolling(w).min()) / c
    f["mom_5"] = f["ret_5"] / f["vol_30"].replace(0, np.nan)          # vol-scaled momentum
    f["mr_z_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std(ddof=0).replace(0, np.nan)
    hr = d["dt"].dt.hour + d["dt"].dt.minute / 60.0
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    # data-quality indicators
    step = d["ts"].diff()
    gap_bar = (step.isna()) | (step != tf)
    f["gap_recent"] = gap_bar.rolling(CONTIGUOUS_WINDOW, min_periods=1).max().astype(float)
    f["bars_since_gap"] = (~gap_bar).astype(int).groupby(gap_bar.cumsum()).cumsum().astype(float)

    feats = [x for x in f.columns if x not in ("ts", "gap_recent", "bars_since_gap")]
    f["valid"] = f[feats].notna().all(axis=1) & np.isfinite(f[feats]).all(axis=1) & \
        (f["bars_since_gap"] >= CONTIGUOUS_WINDOW)
    return f


MODEL_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "bull_run", "bear_run", "body_ratio", "upper_wick",
    "lower_wick", "body_dir", "vol_10", "vol_30", "ema_dist_5", "ema_dist_20", "ema_cross", "rsi_14",
    "macd_hist_n", "macd_n", "bb_pos", "bb_width", "atr_n", "dist_high_10", "dist_low_10",
    "dist_high_30", "dist_low_30", "mom_5", "mr_z_20", "hour_sin", "hour_cos",
]


def build_dataset(df: pd.DataFrame, tf: int = 60, doji_tol: float = 1e-5) -> pd.DataFrame:
    """Features + 'label' ('UP'/'DOWN'/'DOJI'/NaN) + 'y' (1 UP, 0 DOWN, NaN otherwise)."""
    from data.candles import label_next
    d = df.reset_index(drop=True)
    f = build_features(d, tf, doji_tol)
    f["label"] = label_next(d, doji_tol, tf)
    f["y"] = f["label"].map({"UP": 1.0, "DOWN": 0.0})
    return f


def trainable(ds: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for supervised training/evaluation: valid features and a decisive label.
    DOJI rows are excluded from fitting but reported separately by the evaluators."""
    return ds[ds["valid"] & ds["y"].notna()].reset_index(drop=True)
