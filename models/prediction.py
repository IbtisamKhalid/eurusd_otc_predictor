"""Turn the latest COMPLETED candles into a gated prediction."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import Settings
from data.validation import is_stale, lag_seconds
from features.engineering import build_features
from models.training import TrainedModel

UP, DOWN, NO_TRADE = "UP", "DOWN", "NO TRADE"


@dataclass
class Prediction:
    direction: str                      # UP | DOWN | NO TRADE (gated)
    reason: Optional[str]               # why NO TRADE
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    based_on_ts: Optional[int] = None   # last completed bar used
    target_ts: Optional[int] = None     # candle being predicted
    signal_ts: Optional[float] = None   # server-clock time of the signal
    lag_s: Optional[float] = None       # seconds after target open when computed (freshness)
    raw_direction: Optional[str] = None # ungated argmax, for shadow forward-testing only
    p_up: Optional[float] = None        # calibrated only; None otherwise
    p_down: Optional[float] = None
    calibrated: bool = False
    calibration_status: str = ""
    passes_threshold: bool = False
    top_features: list[tuple[str, float]] = field(default_factory=list)


def _explain(tm: TrainedModel, x: pd.DataFrame, k: int = 5) -> list[tuple[str, float]]:
    """Signed contributions to the logit for logistic regression; empty for other models."""
    if tm.name != "logreg":
        return []
    scaler, lr = tm.model.steps[0][1], tm.model.steps[1][1]
    z = scaler.transform(x[tm.features])[0] * lr.coef_[0]
    order = np.argsort(-np.abs(z))[:k]
    return [(tm.features[i], float(z[i])) for i in order]


def make_prediction(candles: pd.DataFrame, tm: Optional[TrainedModel], settings: Settings,
                    server_now: float, connected: bool) -> Prediction:
    """`candles` must contain COMPLETED bars only (the caller guarantees this)."""
    tf = settings.timeframe_s
    if candles is None or len(candles) == 0:
        return Prediction(NO_TRADE, "no candle data available")
    last_ts = int(candles["ts"].iloc[-1])
    target_ts = last_ts + tf
    base = dict(model_name=tm.name if tm else None, model_version=tm.version if tm else None,
                based_on_ts=last_ts, target_ts=target_ts, signal_ts=server_now,
                lag_s=lag_seconds(last_ts, tf, server_now))
    if not connected:
        return Prediction(NO_TRADE, "feed disconnected / last API call failed", **base)
    if is_stale(last_ts, tf, server_now, settings.max_staleness_s):
        return Prediction(NO_TRADE, f"data stale: last completed candle closed {base['lag_s']:.0f}s ago "
                                    f"(limit {settings.max_staleness_s}s)", **base)
    if base["lag_s"] < -5:
        return Prediction(NO_TRADE, "clock skew: newest completed candle is in the future", **base)
    if tm is None:
        return Prediction(NO_TRADE, "no trained model (train one first)", **base)
    f = build_features(candles, tf, settings.doji_tolerance)
    row = f.iloc[[-1]]
    if not bool(row["valid"].iloc[0]):
        return Prediction(NO_TRADE, "insufficient history or recent missing candles in lookback window", **base)
    raw, cal = tm.predict_p_up(row)
    p = float(cal[0])
    calibrated = tm.calibration.ok and tm.calibrator is not None
    raw_dir = UP if p >= 0.5 else DOWN
    out = Prediction(NO_TRADE, None, raw_direction=raw_dir, calibrated=calibrated,
                     calibration_status=tm.calibration.status,
                     top_features=_explain(tm, row), **base)
    if not calibrated:
        out.reason = f"model not calibrated ({tm.calibration.status}); shadow prediction logged only"
        return out
    out.p_up, out.p_down = p, 1 - p
    thr = settings.decision_threshold
    if p >= thr:
        out.direction, out.passes_threshold = UP, True
    elif 1 - p >= thr:
        out.direction, out.passes_threshold = DOWN, True
    else:
        out.reason = f"uncertain: max probability {max(p, 1 - p):.3f} < threshold {thr:.2f}"
    return out
