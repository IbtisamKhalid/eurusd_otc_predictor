import math
import numpy as np
import pandas as pd
import pytest
from config import Settings
from evaluation import metrics as M
from evaluation.backtest import run_backtest, simulate_from_backtest
from evaluation.simulator import SimConfig, break_even_win_rate, simulate
from evaluation.walk_forward import walk_forward_predict
from features.engineering import build_dataset, trainable
from tests.fixtures import make_synthetic_candles


def test_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    pred = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    r = M.classification_report(y, pred, n_total_rows=16)
    assert r["accuracy"] == pytest.approx(6 / 8)
    assert r["confusion"] == {"tp_up": 3, "fn_up": 1, "tn_down": 3, "fp_up": 1}
    assert r["up_precision"] == pytest.approx(3 / 4) and r["up_recall"] == pytest.approx(3 / 4)
    assert r["down_precision"] == pytest.approx(3 / 4) and r["balanced_accuracy"] == pytest.approx(0.75)
    assert r["coverage"] == 0.5 and r["no_trade_frequency"] == 0.5


def test_balanced_accuracy_penalises_majority_guessing():
    y = np.array([1] * 90 + [0] * 10)
    r = M.classification_report(y, np.ones(100, dtype=int))
    assert r["accuracy"] == 0.9 and r["balanced_accuracy"] == 0.5 and r["down_recall"] == 0.0


def test_brier_logloss_ece():
    y, p = np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.6, 0.4])
    assert M.brier(y, p) == pytest.approx((0.01 + 0.01 + 0.16 + 0.16) / 4)
    assert M.log_loss_(y, p) == pytest.approx(-(math.log(.9) + math.log(.9) + math.log(.6) + math.log(.6)) / 4)
    assert M.brier(y, np.full(4, 0.5)) == 0.25
    perfect = np.array([0.0] * 500 + [1.0] * 500), np.array([0.0] * 500 + [1.0] * 500)
    assert M.expected_calibration_error(*perfect) == 0.0


def test_wilson_and_binomial():
    lo, hi = M.wilson_interval(55, 100)
    assert lo < 0.55 < hi and 0.45 < lo < 0.46
    assert M.binomial_p_value_above(5, 10, 0.5) == pytest.approx(0.623046875)
    assert M.wilson_interval(0, 0) == (None, None)
    assert 0.0 < M.binomial_p_value_above(52000, 100000, 0.5) < 1e-30   # no overflow at large n
    assert M.binomial_p_value_above(0, 10, 0.5) == 1.0


def test_walk_forward_is_chronological_and_leak_free():
    ds = trainable(build_dataset(make_synthetic_candles(3000, seed=21, autocorr=0.2), doji_tol=0.0))
    wf = walk_forward_predict(ds, "logreg", min_train=800, test_size=300)
    assert wf["ts"].is_monotonic_increasing and wf["ts"].is_unique
    assert wf["ts"].min() == ds["ts"].iloc[800]
    assert wf.groupby("fold")["ts"].min().is_monotonic_increasing
    # Poison every label AFTER fold 0's test block: fold-0 predictions must not change.
    ds2 = ds.copy()
    ds2.loc[800 + 300:, "y"] = 1 - ds2.loc[800 + 300:, "y"]
    a = walk_forward_predict(ds, "logreg", 800, 300, max_folds=1)
    b = walk_forward_predict(ds2, "logreg", 800, 300, max_folds=1)
    np.testing.assert_allclose(a["p"], b["p"])
    # Poison labels INSIDE the test block: predictions still unchanged (labels never used to predict).
    ds3 = ds.copy()
    ds3.loc[800:1099, "y"] = 1 - ds3.loc[800:1099, "y"]
    c = walk_forward_predict(ds3, "logreg", 800, 300, max_folds=1)
    np.testing.assert_allclose(a["p"], c["p"])


def test_simulator_math_and_tie_rules():
    t = pd.DataFrame({"direction": [1, 1, -1, 1, 1],
                      "t_open": [1.0, 1.0, 1.0, 1.0, 1.0],
                      "t_close": [1.1, 0.9, 0.9, 1.0, 1.1]})
    r = simulate(t, SimConfig(payout_ratio=0.8, stake=10, tie_rule="refund"))
    assert (r["wins"], r["losses"], r["ties"]) == (3, 1, 1)
    assert r["net_pnl"] == pytest.approx(3 * 8 - 10)
    assert r["max_consecutive_losses"] == 1
    assert r["break_even_win_rate"] == pytest.approx(1 / 1.8)
    r2 = simulate(t, SimConfig(payout_ratio=0.8, stake=10, tie_rule="loss"))
    assert r2["net_pnl"] == pytest.approx(24 - 20) and r2["no_signal_baseline_pnl"] == 0.0


def test_simulator_drawdown_streak_and_delay():
    d = [1] * 6
    close = [0.9, 0.9, 0.9, 1.1, 0.9, 0.9]
    r = simulate(pd.DataFrame({"direction": d, "t_open": [1.0] * 6, "t_close": close}),
                 SimConfig(payout_ratio=1.0, stake=1.0))
    assert r["max_consecutive_losses"] == 3 and r["max_drawdown"] == 4.0  # equity 0,-1,-2,-3,-2,-3,-4
    # With a 30 s delay half the move is already gone: entry = 1.05, close 1.1 still wins.
    r = simulate(pd.DataFrame({"direction": [1], "t_open": [1.0], "t_close": [1.1]}),
                 SimConfig(delay_s=30, tol=1e-5))
    assert r["wins"] == 1
    # Entry at close (delay == tf) => zero move => tie.
    r = simulate(pd.DataFrame({"direction": [1], "t_open": [1.0], "t_close": [1.1]}),
                 SimConfig(delay_s=60))
    assert r["ties"] == 1
    assert break_even_win_rate(0.92) == pytest.approx(0.5208, abs=1e-4)
    assert simulate(pd.DataFrame(columns=["direction", "t_open", "t_close"]), SimConfig())["n_trades"] == 0


def test_backtest_reports_insufficient_data():
    r = run_backtest(make_synthetic_candles(500), Settings(), test_size=500)
    assert r["sufficient"] is False and "No predictive-performance claim" in r["message"]


def test_backtest_runs_end_to_end_on_synthetic_data():
    s = Settings(min_train_samples=800)
    candles = make_synthetic_candles(3500, seed=31, autocorr=0.0, gaps_at=(1200,))
    res = run_backtest(candles, s, test_size=400)
    assert res["sufficient"] and set(res["models"]) == {"majority", "momentum", "logreg", "hgb"}
    for name, rep in res["models"].items():
        a = rep["all_rows"]
        assert a["n_predictions"] > 1000 and 0.4 < a["accuracy"] < 0.6
        assert "regimes" in rep and rep["p_value_vs_breakeven"] is not None
    assert "doji_outcomes_in_range" in res and res["date_range_utc"][0] < res["date_range_utc"][1]
    # random-walk data must never be declared an edge
    assert all("NO demonstrated edge" in v for v in res["verdicts"].values())
    sim = simulate_from_backtest(res, "logreg", candles, s, SimConfig(payout_ratio=0.85))
    assert "n_trades" in sim
