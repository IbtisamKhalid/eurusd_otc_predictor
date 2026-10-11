"""Transparent baselines with a scikit-learn-like API (fit / predict_proba, classes_ = [0, 1])."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Always predicts the most frequent training class (Laplace-smoothed prior as probability)."""
    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.p_up_ = (y.sum() + 1) / (len(y) + 2)
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p_up_), np.full(n, self.p_up_)])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MomentumClassifier(BaseEstimator, ClassifierMixin):
    """Predict that the next candle continues the direction of the last candle (body_dir).
    Flat last candle -> falls back to the training majority. Probabilities are NOT meaningful
    (fixed 0.55/0.45 placeholders used only for ranking); never display them."""
    def __init__(self, column: str = "body_dir", reverse: bool = False):
        self.column, self.reverse = column, reverse

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fallback_ = int(np.asarray(y).mean() >= 0.5)
        return self

    def _col(self, X):
        return np.asarray(X[self.column] if isinstance(X, pd.DataFrame) else X[:, 0], dtype=float)

    def predict(self, X):
        s = np.sign(self._col(X))
        if self.reverse:
            s = -s
        return np.where(s > 0, 1, np.where(s < 0, 0, self.fallback_)).astype(int)

    def predict_proba(self, X):
        up = self.predict(X) == 1
        p = np.where(up, 0.55, 0.45)
        return np.column_stack([1 - p, p])
