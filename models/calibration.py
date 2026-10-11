"""Probability calibration fitted ONLY on held-out chronological validation data."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import brier, expected_calibration_error, null_expected_ece

MIN_CALIB_SAMPLES = 400
MAX_EXCESS_ECE = 0.02   # allowed ECE above the pure sampling-noise floor


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class SigmoidCalibrator:
    """Platt scaling: p' = sigmoid(a * logit(p) + b)."""
    def fit(self, p: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


@dataclass
class CalibrationReport:
    status: str                 # "calibrated" | "uncalibrated: <reason>"
    n_val: int
    brier_raw: Optional[float] = None
    brier_cal: Optional[float] = None
    ece_cal_cv: Optional[float] = None
    ece_noise_floor: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.status == "calibrated"


def fit_calibrator(p_val: np.ndarray, y_val: np.ndarray, n_bins: int = 10
                   ) -> tuple[Optional[SigmoidCalibrator], CalibrationReport]:
    """Assess with 2-fold cross-fitting INSIDE the (already out-of-sample) validation block:
    calibrator fitted on one half scores the other half, and vice versa, so every validation
    row gets an out-of-fold calibrated probability (halves the noise of a single split). The
    final calibrator is then refit on the whole block. Probabilities may be displayed only if
    the pooled out-of-fold ECE is within MAX_EXCESS_ECE of the pure sampling-noise floor."""
    n = len(y_val)
    if n < MIN_CALIB_SAMPLES:
        return None, CalibrationReport(f"uncalibrated: only {n} validation rows (< {MIN_CALIB_SAMPLES})", n)
    if len(np.unique(y_val)) < 2:
        return None, CalibrationReport("uncalibrated: validation labels contain a single class", n)
    h = n // 2
    if len(np.unique(y_val[:h])) < 2 or len(np.unique(y_val[h:])) < 2:
        return None, CalibrationReport("uncalibrated: a validation half contains a single class", n)
    oof = np.empty(n)
    oof[h:] = SigmoidCalibrator().fit(p_val[:h], y_val[:h]).transform(p_val[h:])
    oof[:h] = SigmoidCalibrator().fit(p_val[h:], y_val[h:]).transform(p_val[:h])
    n_bins = max(3, min(n_bins, n // 100))
    ece = expected_calibration_error(y_val, oof, n_bins)
    floor = null_expected_ece(oof, n_bins)
    rep = CalibrationReport("calibrated", n, brier(y_val, p_val), brier(y_val, oof), ece, floor)
    if ece > floor + MAX_EXCESS_ECE:
        rep.status = (f"uncalibrated: out-of-fold ECE {ece:.3f} exceeds noise floor {floor:.3f} "
                      f"+ {MAX_EXCESS_ECE}")
    final = SigmoidCalibrator().fit(p_val, y_val)
    return final, rep
