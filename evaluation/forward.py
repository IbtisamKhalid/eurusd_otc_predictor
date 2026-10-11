"""Forward-test bookkeeping: log predictions, then match them to realised candles."""
from __future__ import annotations
from typing import Optional

from config import Settings
from data.candles import classify
from data.database import Database
from models.prediction import NO_TRADE, Prediction


def record_prediction(db: Database, pred: Prediction, s: Settings) -> Optional[int]:
    """Idempotent per (target candle, model). Only predictions for a real target are stored."""
    if pred.model_name is None or pred.target_ts is None:
        return None
    return db.insert_prediction(
        venue=s.venue, symbol=s.symbol, tf=s.timeframe_s, target_ts=pred.target_ts,
        signal_ts=pred.signal_ts, based_on_ts=pred.based_on_ts, model_name=pred.model_name,
        model_version=pred.model_version, direction=pred.direction, raw_direction=pred.raw_direction,
        p_up=pred.p_up, p_down=pred.p_down, calibrated=int(pred.calibrated), lag_s=pred.lag_s,
        reason=pred.reason)


def match_outcomes(db: Database, s: Settings) -> int:
    """Resolve predictions whose target candle is now stored (completed). Never double-counts."""
    candles = db.load_candles(s.venue, s.symbol, s.timeframe_s)
    if candles.empty:
        return 0
    by_ts = candles.set_index("ts")
    n = 0
    for p in db.unresolved_predictions():
        if p["target_ts"] not in by_ts.index:
            continue
        r = by_ts.loc[p["target_ts"]]
        out = classify(float(r["open"]), float(r["close"]), s.doji_tolerance)
        raw_c = None if out == "DOJI" or p["raw_direction"] is None else int(p["raw_direction"] == out)
        sig_c = None if out == "DOJI" or p["direction"] == NO_TRADE else int(p["direction"] == out)
        n += int(db.insert_outcome(p["id"], out, float(r["open"]), float(r["close"]), raw_c, sig_c))
    return n
