import numpy as np
import pandas as pd
import pytest
from features import indicators as ind
from features.engineering import MODEL_FEATURES, build_dataset, build_features, trainable
from tests.fixtures import make_synthetic_candles


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up).dropna().iloc[-1] == 100.0
    assert ind.rsi(-up + 100).dropna().iloc[-1] < 1e-9
    r = ind.rsi(pd.Series(np.random.default_rng(1).normal(size=300).cumsum() + 50)).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert ind.rsi(pd.Series([5.0] * 50)).dropna().iloc[-1] == 50.0


def test_ema_matches_manual():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    e = ind.ema(s, 3)
    assert e.iloc[:2].isna().all()
    a = 2 / 4
    manual = [1.0]
    for x in s.iloc[1:]:
        manual.append(a * x + (1 - a) * manual[-1])
    assert e.iloc[-1] == pytest.approx(manual[-1])


def test_macd_identity():
    c = pd.Series(np.random.default_rng(2).normal(size=200).cumsum() + 100)
    m = ind.macd(c).dropna()
    assert (m["macd_hist"] - (m["macd"] - m["macd_signal"])).abs().max() < 1e-12


def test_bollinger_ordering_and_position():
    c = pd.Series(np.random.default_rng(3).normal(size=100).cumsum() + 100)
    b = ind.bollinger(c).dropna()
    assert (b["bb_upper"] >= b["bb_mid"]).all() and (b["bb_mid"] >= b["bb_lower"]).all()
    assert (ind.bollinger(c)["bb_pos"].dropna().between(-0.5, 1.5)).all()


def test_atr_non_negative_and_true_range():
    df = make_synthetic_candles(100)
    tr = ind.true_range(df)
    assert (tr.dropna() >= (df["high"] - df["low"]).iloc[1:] - 1e-15).all()
    assert (ind.atr(df).dropna() >= 0).all()


def test_run_length():
    f = pd.Series([True, True, False, True, True, True, False])
    assert list(ind.run_length(f)) == [1, 2, 0, 1, 2, 3, 0]


def test_features_have_no_lookahead_truncation_invariance():
    df = make_synthetic_candles(400, seed=5)
    full = build_features(df)
    for k in (80, 150, 250, 399):
        part = build_features(df.iloc[: k + 1].reset_index(drop=True))
        a, b = full.iloc[: k + 1].reset_index(drop=True), part
        cols = [c for c in a.columns if c != "valid"]
        pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-12)
        assert (a["valid"] == b["valid"]).all()


def test_poisoning_future_bars_does_not_change_past_features():
    df = make_synthetic_candles(300, seed=6)
    base = build_features(df)
    df2 = df.copy()
    df2.loc[200:, ["open", "high", "low", "close"]] *= 1.5     # wild future values
    poisoned = build_features(df2)
    cols = [c for c in base.columns if c != "valid"]
    pd.testing.assert_frame_equal(base.loc[:199, cols], poisoned.loc[:199, cols])


def test_label_is_next_bar_and_features_exclude_it():
    df = make_synthetic_candles(300, seed=7)
    ds = build_dataset(df, doji_tol=0.0)
    t = 120
    nxt = df.iloc[t + 1]
    expect = "UP" if nxt["close"] > nxt["open"] else "DOWN"
    assert ds.loc[t, "label"] == expect
    # change only bar t+1's close: features of row t unchanged, label may change
    df2 = df.copy()
    df2.loc[t + 1, "close"] = df2.loc[t + 1, "open"] * (0.5 if expect == "UP" else 1.5)
    ds2 = build_dataset(df2, doji_tol=0.0)
    cols = [c for c in MODEL_FEATURES]
    assert (ds.loc[t, cols] == ds2.loc[t, cols]).all()
    assert ds2.loc[t, "label"] != expect


def test_gap_invalidates_rows_and_sets_flags():
    df = make_synthetic_candles(300, seed=8, gaps_at=(150,))
    ds = build_dataset(df)
    gap_row = 150                       # first row after the removed bar
    assert (df.loc[gap_row, "ts"] - df.loc[gap_row - 1, "ts"]) == 120
    assert not ds.loc[gap_row:gap_row + 29, "valid"].any()
    assert ds.loc[gap_row + 31, "valid"]
    assert ds.loc[gap_row, "gap_recent"] == 1.0
    assert pd.isna(ds.loc[gap_row - 1, "label"])   # next bar isn't the next minute


def test_trainable_excludes_doji_and_invalid():
    df = make_synthetic_candles(500, seed=9)
    ds = build_dataset(df, doji_tol=5e-5)
    assert (ds["label"] == "DOJI").any()
    tr = trainable(ds)
    assert tr["y"].isin([0.0, 1.0]).all() and tr["valid"].all()
    assert tr[MODEL_FEATURES].notna().all().all()
