"""Model factory and chronological train/calibrate procedure."""
from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.engineering import MODEL_FEATURES
from models.baseline import MajorityClassifier, MomentumClassifier
from models.calibration import CalibrationReport, fit_calibrator

MODEL_NAMES = ("majority", "momentum", "logreg", "hgb")
EMBARGO_ROWS = 15


def make_model(name: str):
    if name == "majority":
        return MajorityClassifier()
    if name == "momentum":
        return MomentumClassifier()
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if name == "hgb":  # strongly regularised: weak-signal, noisy target
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                              min_samples_leaf=100, l2_regularization=1.0,
                                              random_state=0)
    raise ValueError(f"unknown model {name!r}")


def calibratable(name: str) -> bool:
    return name in ("logreg", "hgb")


@dataclass
class TrainedModel:
    name: str
    version: str
    model: Any
    calibrator: Optional[Any]
    calibration: CalibrationReport
    features: list[str]
    n_train: int
    trained_at: float
    train_end_ts: int
    info: dict = field(default_factory=dict)

    def predict_p_up(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.predict_proba(X[self.features])[:, 1]
        cal = self.calibrator.transform(raw) if self.calibrator is not None else raw
        return raw, cal


def train_model(ds: pd.DataFrame, name: str, min_samples: int = 2000, calib_frac: float = 0.2,
                embargo: int = EMBARGO_ROWS) -> TrainedModel:
    """`ds` = engineering.trainable(...) output, chronological. Fit on the earlier part,
    calibrate on the later part (separated by an embargo)."""
    n = len(ds)
    if n < min_samples:
        raise ValueError(f"Insufficient data: {n} usable rows < minimum {min_samples}. "
                         f"Collect more history (see README).")
    n_cal = int(n * calib_frac)
    fit_end = n - n_cal - embargo
    X, y = ds[MODEL_FEATURES], ds["y"].astype(int).to_numpy()
    model = make_model(name)
    model.fit(X.iloc[:fit_end], y[:fit_end])
    calib, report = None, CalibrationReport("uncalibrated: baseline model has no probabilities", 0)
    if calibratable(name):
        p_val = model.predict_proba(X.iloc[fit_end + embargo:])[:, 1]
        calib, report = fit_calibrator(p_val, y[fit_end + embargo:])
    # final refit on everything before the calibration block is NOT done: the calibrator
    # was fitted to this exact fitted model, so we keep it as-is.
    end_ts = int(ds["ts"].iloc[-1])
    version = f"{name}-{end_ts}-{hashlib.sha1(f'{n}:{end_ts}:{name}'.encode()).hexdigest()[:6]}"
    return TrainedModel(name, version, model, calib, report, list(MODEL_FEATURES), fit_end,
                        time.time(), end_ts, {"rows_total": n, "rows_fit": fit_end, "rows_calibration": n_cal})


def save_model(tm: TrainedModel, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{tm.name}.joblib"
    joblib.dump(tm, p)
    return p


def load_model(directory: Path, name: str) -> Optional[TrainedModel]:
    p = directory / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None
