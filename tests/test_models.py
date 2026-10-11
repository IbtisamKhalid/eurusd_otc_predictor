import numpy as np
import pandas as pd
import pytest
from config import Settings
from features.engineering import MODEL_FEATURES, build_dataset, trainable
from models.baseline import MajorityClassifier, MomentumClassifier
from models.calibration import MIN_CALIB_SAMPLES, fit_calibrator
from models.prediction import make_prediction
from models.training import TrainedModel, train_model
from evaluation.metrics import expected_calibration_error
from tests.fixtures import make_synthetic_candles


def test_majority_and_momentum_baselines():
    X = pd.DataFrame({"body_dir": [1.0, -1.0, 0.0, 1.0]})
    m = MajorityClassifier().fit(X, [1, 1, 1, 0])
    assert (m.predict(X) == 1).all() and m.predict_proba(X).shape == (4, 2)
    mom = MomentumClassifier().fit(X, [0, 0, 0, 1])
    assert list(mom.predict(X)) == [1, 0, 0, 1]           # flat -> training majority (0)
    assert mom.predict_proba(X)[0, 1] > 0.5
    assert list(MomentumClassifier(reverse=True).fit(X, [1, 1, 1, 0]).predict(X)) == [0, 1, 1, 0]


def test_logreg_finds_signal_when_it_exists_and_not_when_it_does_not():
    def acc(autocorr, seed):
        ds = trainable(build_dataset(make_synthetic_candles(5000, seed=seed, autocorr=autocorr), doji_tol=0.0))
        cut = int(len(ds) * 0.7)
        tm = train_model(ds.iloc[:cut].reset_index(drop=True), "logreg", min_samples=500)
        te = ds.iloc[cut + 20:]
        raw, _ = tm.predict_p_up(te)
        return float(((raw >= 0.5).astype(int) == te["y"].astype(int)).mean())
    assert acc(0.25, 1) > 0.56            # planted AR(1) structure is detectable
    assert abs(acc(0.0, 2) - 0.5) < 0.04  # pure random walk: no spurious edge


def test_train_requires_minimum_samples():
    ds = trainable(build_dataset(make_synthetic_candles(400, seed=3)))
    with pytest.raises(ValueError):
        train_model(ds, "logreg", min_samples=2000)


def test_calibrator_refuses_small_or_single_class():
    p = np.random.default_rng(0).uniform(0.4, 0.6, 100)
    cal, rep = fit_calibrator(p, (p > 0.5).astype(int))
    assert cal is None and not rep.ok and "validation rows" in rep.status
    cal, rep = fit_calibrator(np.full(1000, 0.5), np.ones(1000, dtype=int))
    assert cal is None and "single class" in rep.status


def test_calibrator_fixes_overconfident_scores():
    rng = np.random.default_rng(1)
    n = 6000
    true_p = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    overconf = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))     # distorted scores
    assert expected_calibration_error(y, overconf) > 0.05
    cal, rep = fit_calibrator(overconf, y)
    assert cal is not None and rep.ok and rep.ece_cal_cv < rep.ece_noise_floor + 0.02
    assert rep.brier_cal <= rep.brier_raw


def test_calibration_gate_rejects_unstable_relationship():
    rng = np.random.default_rng(5)
    n = 4000
    p = rng.uniform(0.2, 0.8, n)
    y = np.where(np.arange(n) < n // 2, rng.uniform(size=n) < p, rng.uniform(size=n) < 1 - p).astype(int)
    cal, rep = fit_calibrator(p, y)
    assert not rep.ok and "exceeds noise floor" in rep.status


def test_calibration_gate_is_not_fooled_by_noise_alone():
    """A perfectly calibrated model judged on a small block must still pass."""
    passes = 0
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        p = rng.uniform(0.4, 0.6, 800)
        y = (rng.uniform(size=800) < p).astype(int)
        passes += fit_calibrator(p, y)[1].ok
    assert passes >= 8


def _trained(calibrated=True):
    ds = trainable(build_dataset(make_synthetic_candles(5000, seed=11, autocorr=0.3), doji_tol=0.0))
    tm = train_model(ds, "logreg", min_samples=500)
    if not calibrated:
        tm.calibrator, tm.calibration.status = None, "uncalibrated: forced"
    return tm


def _candles(n=300):
    return make_synthetic_candles(n, seed=12, autocorr=0.3)


def test_prediction_gating():
    s = Settings()
    tm, df = _trained(), _candles()
    last = int(df["ts"].iloc[-1])
    fresh = last + 60 + 10
    p = make_prediction(df, tm, s, fresh, connected=True)
    assert p.target_ts == last + 60 and p.based_on_ts == last
    assert p.calibrated and p.p_up is not None and abs(p.p_up + p.p_down - 1) < 1e-9
    assert p.direction in ("UP", "DOWN", "NO TRADE") and p.lag_s == 10
    assert p.top_features
    # disconnected
    p = make_prediction(df, tm, s, fresh, connected=False)
    assert p.direction == "NO TRADE" and "disconnected" in p.reason
    # stale
    p = make_prediction(df, tm, s, last + 60 + 1000, connected=True)
    assert p.direction == "NO TRADE" and "stale" in p.reason
    # no model
    p = make_prediction(df, None, s, fresh, connected=True)
    assert p.direction == "NO TRADE" and "model" in p.reason
    # no data
    assert make_prediction(df.iloc[0:0], tm, s, fresh, True).direction == "NO TRADE"
    # too little history
    assert "insufficient" in make_prediction(df.iloc[:20], tm, s, int(df['ts'].iloc[19]) + 70, True).reason


def test_uncalibrated_model_hides_probabilities_and_does_not_signal():
    s = Settings()
    tm, df = _trained(calibrated=False), _candles()
    p = make_prediction(df, tm, s, int(df["ts"].iloc[-1]) + 70, connected=True)
    assert p.direction == "NO TRADE" and p.p_up is None and p.p_down is None
    assert p.raw_direction in ("UP", "DOWN") and "not calibrated" in p.reason


def test_threshold_gate():
    tm, df = _trained(), _candles()
    now = int(df["ts"].iloc[-1]) + 70
    hi = make_prediction(df, tm, Settings(decision_threshold=0.95), now, True)
    assert hi.direction == "NO TRADE" and "uncertain" in hi.reason
    lo = make_prediction(df, tm, Settings(decision_threshold=0.5), now, True)
    assert lo.direction in ("UP", "DOWN") and lo.passes_threshold


def test_prediction_ignores_forming_candle_by_contract():
    """The caller passes completed bars only; appending a *future* bar must change the
    target timestamp, proving the target is always last_complete + tf."""
    tm, s = _trained(), Settings()
    df = _candles(300)
    p1 = make_prediction(df.iloc[:-1], tm, s, int(df["ts"].iloc[-2]) + 70, True)
    assert p1.target_ts == int(df["ts"].iloc[-1])
