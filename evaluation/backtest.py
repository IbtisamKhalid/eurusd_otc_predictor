"""Leakage-safe historical backtest: walk-forward predictions, metrics, baselines, simulator."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from config import Settings
from evaluation import metrics as M
from evaluation.simulator import SimConfig, break_even_win_rate, simulate
from evaluation.walk_forward import walk_forward_predict
from features.engineering import build_dataset, trainable
from models.training import MODEL_NAMES


def _slice_report(wf: pd.DataFrame, thr: float, total_rows: int, use_cal: bool) -> dict:
    p = wf["p"].to_numpy()
    pred = (p >= 0.5).astype(int)
    out = {"all_rows": M.classification_report(wf["y"], pred, p if use_cal else None, total_rows)}
    if use_cal:
        mask = (np.maximum(p, 1 - p) >= thr) & wf["calibrated"].to_numpy()
        sub = wf[mask]
        out["thresholded"] = M.classification_report(sub["y"], (sub["p"] >= 0.5).astype(int),
                                                     sub["p"].to_numpy() if len(sub) else None, total_rows) \
            if len(sub) else {"n_predictions": 0, "coverage": 0.0, "no_trade_frequency": 1.0}
    return out


def _regimes(wf: pd.DataFrame, tr: pd.DataFrame) -> dict:
    j = wf.merge(tr[["ts", "vol_30", "hour_sin"]].assign(
        hour=lambda d: pd.to_datetime(d["ts"], unit="s", utc=True).dt.hour), on="ts", how="left")
    j["correct"] = ((j["p"] >= 0.5).astype(int) == j["y"]).astype(int)
    out = {}
    j["session"] = pd.cut(j["hour"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
    out["by_utc_session"] = {str(k): {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
                             for k, g in j.groupby("session", observed=True)}
    if j["vol_30"].nunique() > 3:
        j["vol_bucket"] = pd.qcut(j["vol_30"], 3, labels=["low", "mid", "high"])
        out["by_volatility"] = {str(k): {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
                                for k, g in j.groupby("vol_bucket", observed=True)}
    j["block"] = pd.qcut(j["ts"], min(5, max(1, len(j) // 200)), labels=False, duplicates="drop")
    out["by_time_block"] = {str(int(k)): {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
                            for k, g in j.groupby("block")}
    return out


def run_backtest(candles: pd.DataFrame, s: Settings, model_names: Sequence[str] = MODEL_NAMES,
                 test_size: int = 500, max_folds: Optional[int] = None) -> dict:
    ds = build_dataset(candles, s.timeframe_s, s.doji_tolerance)
    tr = trainable(ds)
    min_train = s.min_train_samples
    if len(tr) < min_train + test_size:
        return {"sufficient": False, "n_usable_rows": int(len(tr)),
                "message": f"Insufficient data: {len(tr)} usable rows; need >= {min_train + test_size} "
                           f"(min_train {min_train} + one test block of {test_size}). "
                           "No predictive-performance claim can be made."}
    res: dict = {"sufficient": True, "n_usable_rows": int(len(tr)), "models": {}, "settings": {
        "min_train": min_train, "test_size": test_size, "threshold": s.decision_threshold}}
    be = break_even_win_rate(s.payout_ratio)
    first_ts = last_ts = None
    for name in model_names:
        wf = walk_forward_predict(tr, name, min_train, test_size, max_folds=max_folds)
        if wf.empty:
            continue
        first_ts, last_ts = int(wf["ts"].min()), int(wf["ts"].max())
        use_cal = bool(wf["calibrated"].any())
        # only rows from folds that passed calibration carry displayable probabilities
        rep = _slice_report(wf, s.decision_threshold, len(wf), use_cal and wf["calibrated"].all())
        k, n = round(rep["all_rows"]["accuracy"] * len(wf)), len(wf)
        rep["p_value_vs_50pct"] = M.binomial_p_value_above(k, n, 0.5)
        rep["p_value_vs_breakeven"] = M.binomial_p_value_above(k, n, be)
        rep["calibrated_folds_fraction"] = float(wf.groupby("fold")["calibrated"].first().mean())
        rep["regimes"] = _regimes(wf, tr)
        res["models"][name] = rep
        res.setdefault("_wf", {})[name] = wf
    if first_ts is None:
        res["sufficient"] = False
        res["message"] = "No walk-forward folds could be formed."
        return res
    span = ds[(ds["ts"] >= first_ts) & (ds["ts"] <= last_ts)]
    res["date_range_utc"] = [pd.to_datetime(first_ts, unit="s", utc=True).isoformat(),
                             pd.to_datetime(last_ts, unit="s", utc=True).isoformat()]
    res["doji_outcomes_in_range"] = int((span["label"] == "DOJI").sum())
    res["skipped_rows_gaps_or_invalid"] = int((~span["valid"] | span["label"].isna()).sum())
    res["break_even_win_rate_at_payout"] = be
    # verdict vs. baselines
    maj = res["models"].get("majority", {}).get("all_rows", {}).get("accuracy")
    mom = res["models"].get("momentum", {}).get("all_rows", {}).get("accuracy")
    verdicts = {}
    for name, rep in res["models"].items():
        a = rep["all_rows"]
        lo = a["accuracy_ci95"][0]
        beats = lambda b: b is not None and lo is not None and lo > b
        verdicts[name] = ("evidence of edge over break-even" if lo is not None and lo > be else
                          "NO demonstrated edge over break-even (95% CI includes it or is below it)")
        if name not in ("majority", "momentum"):
            verdicts[name] += "; " + ("beats" if beats(max(maj or 0, mom or 0)) else "does not clearly beat") + \
                " the best baseline"
    res["verdicts"] = verdicts
    return res


def to_trades(wf: pd.DataFrame, candles: pd.DataFrame, s: Settings, thr: float) -> pd.DataFrame:
    """Join thresholded walk-forward signals to the target candle's O/C. Row ts -> target ts+tf."""
    sig = wf[(np.maximum(wf["p"], 1 - wf["p"]) >= thr) & wf["calibrated"]].copy()
    sig["target_ts"] = sig["ts"] + s.timeframe_s
    sig["direction"] = np.where(sig["p"] >= 0.5, 1, -1)
    j = sig.merge(candles[["ts", "open", "close"]].rename(columns={"ts": "target_ts", "open": "t_open",
                                                                    "close": "t_close"}),
                  on="target_ts", how="inner")
    return j[["target_ts", "direction", "t_open", "t_close"]]


def simulate_from_backtest(res: dict, name: str, candles: pd.DataFrame, s: Settings, cfg: SimConfig) -> dict:
    wf = res["_wf"][name]
    return simulate(to_trades(wf, candles, s, s.decision_threshold), cfg)


def summary_for_storage(res: dict) -> dict:
    """JSON-safe copy without the raw walk-forward frames."""
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items() if k != "_wf"}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, float) and x != x:
            return None
        return x
    return conv(res)
