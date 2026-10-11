"""Expanding-window walk-forward evaluation with embargo and per-fold calibration."""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from features.engineering import MODEL_FEATURES
from models.calibration import fit_calibrator
from models.training import EMBARGO_ROWS, calibratable, make_model


def walk_forward_predict(ds: pd.DataFrame, name: str, min_train: int = 2000, test_size: int = 500,
                         calib_frac: float = 0.25, embargo: int = EMBARGO_ROWS,
                         max_folds: Optional[int] = None) -> pd.DataFrame:
    """`ds` = engineering.trainable(...). For every fold the model sees only rows strictly
    before the test block minus an embargo; preprocessing (scaler) and the calibrator are fitted
    inside the fold. Returns one row per out-of-sample prediction."""
    X, y = ds[MODEL_FEATURES], ds["y"].astype(int).to_numpy()
    n = len(ds)
    rows = []
    fold, start = 0, min_train
    while start < n and (max_folds is None or fold < max_folds):
        end = min(start + test_size, n)
        train_end = start - embargo
        n_cal = int((train_end) * calib_frac) if calibratable(name) else 0
        fit_end = train_end - n_cal - (embargo if n_cal else 0)
        if fit_end < 200:
            start += test_size
            continue
        model = make_model(name)
        model.fit(X.iloc[:fit_end], y[:fit_end])
        cal, status_ok = None, False
        if n_cal:
            p_val = model.predict_proba(X.iloc[fit_end + embargo:train_end])[:, 1]
            cal, rep = fit_calibrator(p_val, y[fit_end + embargo:train_end])
            status_ok = rep.ok
        raw = model.predict_proba(X.iloc[start:end])[:, 1]
        p = cal.transform(raw) if cal is not None else raw
        rows.append(pd.DataFrame({"ts": ds["ts"].iloc[start:end].to_numpy(), "y": y[start:end],
                                  "p_raw": raw, "p": p, "fold": fold, "calibrated": status_ok}))
        fold += 1
        start = end
    cols = ["ts", "y", "p_raw", "p", "fold", "calibrated"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
