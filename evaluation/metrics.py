"""Performance metrics. All functions take y as {0=DOWN, 1=UP} arrays."""
from __future__ import annotations
import math
from typing import Optional

import numpy as np


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {"tp_up": int(((y == 1) & (pred == 1)).sum()), "fn_up": int(((y == 1) & (pred == 0)).sum()),
            "tn_down": int(((y == 0) & (pred == 0)).sum()), "fp_up": int(((y == 0) & (pred == 1)).sum())}


def _div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def brier(y: np.ndarray, p_up: np.ndarray) -> float:
    return float(np.mean((np.asarray(p_up, float) - np.asarray(y, float)) ** 2))


def log_loss_(y: np.ndarray, p_up: np.ndarray) -> float:
    p = np.clip(np.asarray(p_up, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y: np.ndarray, p_up: np.ndarray, n_bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p_up, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "observed_up": float(y[m].mean())})
    return rows


def expected_calibration_error(y: np.ndarray, p_up: np.ndarray, n_bins: int = 10) -> float:
    rows = calibration_table(y, p_up, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["mean_pred"] - r["observed_up"]) for r in rows) / n) if n else float("nan")


def classification_report(y: np.ndarray, pred: np.ndarray, p_up: Optional[np.ndarray] = None,
                          n_total_rows: Optional[int] = None) -> dict:
    """Metrics over rows where a directional prediction was made."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    n = len(y)
    cm = confusion(y, pred)
    correct = int((y == pred).sum())
    rec_up = _div(cm["tp_up"], cm["tp_up"] + cm["fn_up"])
    rec_dn = _div(cm["tn_down"], cm["tn_down"] + cm["fp_up"])
    bal = None if rec_up is None or rec_dn is None else (rec_up + rec_dn) / 2
    lo, hi = wilson_interval(correct, n)
    rep = {"n_predictions": n, "accuracy": _div(correct, n), "accuracy_ci95": [lo, hi],
           "balanced_accuracy": bal,
           "up_precision": _div(cm["tp_up"], cm["tp_up"] + cm["fp_up"]), "up_recall": rec_up,
           "down_precision": _div(cm["tn_down"], cm["tn_down"] + cm["fn_up"]), "down_recall": rec_dn,
           "confusion": cm, "base_rate_up": float(y.mean()) if n else None}
    if p_up is not None and n:
        rep.update({"brier": brier(y, p_up), "log_loss": log_loss_(y, p_up),
                    "ece": expected_calibration_error(y, p_up),
                    "calibration": calibration_table(y, p_up)})
    if n_total_rows:
        rep["coverage"] = n / n_total_rows
        rep["no_trade_frequency"] = 1 - n / n_total_rows
    return rep


def binomial_p_value_above(k: int, n: int, p0: float = 0.5) -> Optional[float]:
    """One-sided exact P(X >= k | n, p0) -- is accuracy distinguishable from chance/break-even?
    Computed in log space so large n does not overflow."""
    if n == 0:
        return None
    if k <= 0:
        return 1.0
    lp, lq = math.log(p0), math.log(1 - p0)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
            for i in range(k, n + 1)]
    m = max(logs)
    return float(min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs)))


def null_expected_ece(p_up: np.ndarray, n_bins: int = 10) -> float:
    """Expected ECE of a PERFECTLY calibrated model given only sampling noise, for these
    predictions and bin layout. Used so small validation sets are not judged against a bar
    that noise alone would fail."""
    p = np.asarray(p_up, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total, n = 0.0, len(p)
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb:
            sd = math.sqrt(float(np.mean(p[m] * (1 - p[m]))) / nb)
            total += nb * sd * math.sqrt(2 / math.pi)
    return total / n if n else float("nan")
