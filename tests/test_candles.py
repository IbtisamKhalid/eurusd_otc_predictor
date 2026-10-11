import time
import numpy as np
import pandas as pd
import pytest
from api.schemas import Candle
from config import Settings
from data.candles import classify, is_complete, label_next, split_complete, to_frame
from data.database import Database
from data.validation import clean_candles, is_stale, lag_seconds, ohlc_problem
from evaluation.forward import match_outcomes, record_prediction
from models.prediction import Prediction

T0 = 1_700_000_040 - 1_700_000_040 % 60


def C(i, o=1.0, h=1.1, l=0.9, c=1.05, v=1.0):
    return Candle(T0 + 60 * i, o, h, l, c, v)


def test_ohlc_consistency():
    assert ohlc_problem(C(0)) is None
    assert ohlc_problem(C(0, h=1.0, c=1.05)) is not None       # high below close
    assert ohlc_problem(C(0, l=1.0, o=0.95)) is not None       # low above open
    assert ohlc_problem(C(0, o=-1)) is not None
    assert ohlc_problem(Candle(T0, 1, 1.1, 0.9, 1, -2)) is not None


def test_clean_duplicates_order_alignment_gaps():
    cs = [C(2), C(0), C(1), C(1), C(1, c=1.06), Candle(T0 + 61, 1, 1.1, 0.9, 1.0), C(5), C(3, h=0.5)]
    clean, issues = clean_candles(cs)
    kinds = [i.kind for i in issues]
    assert [c.ts for c in clean] == [T0, T0 + 60, T0 + 120, T0 + 300]
    assert clean[1].close == 1.06                       # conflict: kept last
    for k in ("out_of_order", "duplicate", "duplicate_conflict", "misaligned", "invalid_ohlc", "gap"):
        assert k in kinds, k
    gap = [i for i in issues if i.kind == "gap"][0]
    assert "2 bar(s)" in gap.detail


def test_completeness_never_includes_forming_bar():
    cs = [C(0), C(1), C(2)]
    now = T0 + 120 + 30                  # bar 2 is 30 s old -> forming
    done, forming = split_complete(cs, now)
    assert [c.ts for c in done] == [T0, T0 + 60] and [c.ts for c in forming] == [T0 + 120]
    assert is_complete(cs[2], T0 + 180) and not is_complete(cs[2], T0 + 179.9)


def test_freshness():
    assert lag_seconds(T0, 60, T0 + 70) == 10
    assert not is_stale(T0, 60, T0 + 70, 150)
    assert is_stale(T0, 60, T0 + 300, 150)
    assert is_stale(None, 60, T0, 150)


def test_classify_and_doji_tolerance():
    assert classify(1.0, 1.0001, 1e-5) == "UP"
    assert classify(1.0, 0.9999, 1e-5) == "DOWN"
    assert classify(1.0, 1.000005, 1e-5) == "DOJI"
    assert classify(1.0, 1.0, 0.0) == "DOJI"


def test_label_next_alignment_and_gap():
    cs = [C(0), C(1, o=1, c=1.2), C(2, o=1, c=0.8), C(3, o=1, c=1.0), C(5, o=1, c=1.5)]
    df = to_frame(cs)
    lab = label_next(df, 1e-5)
    assert list(lab.iloc[:3]) == ["UP", "DOWN", "DOJI"]
    assert pd.isna(lab.iloc[3])           # next bar is 2 minutes later -> not 'the next candle'
    assert pd.isna(lab.iloc[4])           # no following bar


def test_database_unique_and_no_overwrite():
    db = Database(":memory:")
    n, conf = db.insert_candles("iq", "S", 60, [C(0), C(1)])
    assert n == 2 and conf == []
    n, conf = db.insert_candles("iq", "S", 60, [C(1), C(2)])
    assert n == 1 and conf == []                          # identical redelivery ignored
    n, conf = db.insert_candles("iq", "S", 60, [C(1, c=1.09)])
    assert n == 0 and conf == [T0 + 60]                   # revision reported, original kept
    df = db.load_candles("iq", "S", 60)
    assert len(df) == 3 and df.loc[1, "close"] == 1.05


def test_prediction_to_outcome_matching_and_no_duplicates():
    s = Settings()
    db = Database(":memory:")
    base = dict(model_name="logreg", model_version="v1", based_on_ts=T0, signal_ts=T0 + 65, lag_s=5.0)
    up = Prediction("UP", None, target_ts=T0 + 60, raw_direction="UP", p_up=0.6, p_down=0.4, calibrated=True, **base)
    dn_doji = Prediction("DOWN", None, target_ts=T0 + 120, raw_direction="DOWN", p_up=0.4, p_down=0.6, calibrated=True, **base)
    nt = Prediction("NO TRADE", "uncertain", target_ts=T0 + 180, raw_direction="UP", **base)
    ids = [record_prediction(db, p, s) for p in (up, dn_doji, nt)]
    assert all(i is not None for i in ids)
    assert record_prediction(db, up, s) is None           # duplicate prevented
    assert match_outcomes(db, s) == 0                     # target candles not yet stored
    db.insert_candles(s.venue, s.symbol, 60, [
        Candle(T0 + 60, 1.0, 1.1, 0.9, 1.05), Candle(T0 + 120, 1.0, 1.1, 0.9, 1.0),
        Candle(T0 + 180, 1.0, 1.1, 0.9, 0.95)])
    assert match_outcomes(db, s) == 3
    assert match_outcomes(db, s) == 0                     # idempotent
    df = db.predictions_with_outcomes().set_index("target_ts")
    assert df.loc[T0 + 60, "outcome"] == "UP" and df.loc[T0 + 60, "signal_correct"] == 1
    assert df.loc[T0 + 120, "outcome"] == "DOJI" and pd.isna(df.loc[T0 + 120, "signal_correct"])
    assert df.loc[T0 + 180, "outcome"] == "DOWN" and pd.isna(df.loc[T0 + 180, "signal_correct"])
    assert df.loc[T0 + 180, "raw_correct"] == 0           # shadow prediction UP was wrong


def test_settings_cannot_bypass_safety():
    s = Settings(poll_interval_s=1, max_staleness_s=99999, decision_threshold=0.1, min_train_samples=5)
    assert s.poll_interval_s >= 30 and s.max_staleness_s <= 600
    assert s.decision_threshold >= 0.5 and s.min_train_samples >= 300
