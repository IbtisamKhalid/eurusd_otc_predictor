"""Offline hypothetical fixed-stake simulator. NOT real-money results.

Limits (by design, stated in the UI/README): candle data cannot tell us the broker's actual entry
or settlement price; `delay_s` is a crude linear O->C interpolation of the entry price.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    payout_ratio: float = 0.85      # profit per 1 staked on a win (0.85 = 85%)
    stake: float = 1.0
    delay_s: float = 0.0            # execution delay after target candle opens (0..tf)
    tie_rule: str = "refund"        # 'refund' (stake returned) or 'loss'
    tf: int = 60
    tol: float = 1e-5


def break_even_win_rate(payout_ratio: float, tie_rule: str = "refund") -> float:
    """Win rate among decided (non-tie) trades needed for zero expected P/L."""
    return 1.0 / (1.0 + payout_ratio)


def simulate(trades: pd.DataFrame, cfg: SimConfig) -> dict:
    """`trades`: columns direction (+1 UP / -1 DOWN), t_open, t_close (target candle O/C)."""
    if trades.empty:
        return {"n_trades": 0, "note": "no trades"}
    frac = min(max(cfg.delay_s / cfg.tf, 0.0), 1.0)
    entry = trades["t_open"] + frac * (trades["t_close"] - trades["t_open"])
    move = trades["t_close"] - entry
    tie = move.abs() <= cfg.tol
    win = (np.sign(move) == trades["direction"]) & ~tie
    pnl = np.where(win, cfg.stake * cfg.payout_ratio,
                   np.where(tie, 0.0 if cfg.tie_rule == "refund" else -cfg.stake, -cfg.stake))
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    losing = ~win & ~(tie & (cfg.tie_rule == "refund"))
    streak = best = 0
    for L in losing:
        streak = streak + 1 if L else 0
        best = max(best, streak)
    n_dec = int((~tie).sum())
    return {"n_trades": int(len(pnl)), "wins": int(win.sum()), "ties": int(tie.sum()),
            "losses": int((~win & ~tie).sum()),
            "win_rate_decided": float(win.sum() / n_dec) if n_dec else None,
            "break_even_win_rate": break_even_win_rate(cfg.payout_ratio, cfg.tie_rule),
            "net_pnl": float(eq[-1]), "pnl_per_trade": float(eq[-1] / len(pnl)),
            "max_drawdown": float((peak - eq).max()), "max_consecutive_losses": int(best),
            "no_signal_baseline_pnl": 0.0,
            "disclaimer": "Hypothetical; candle close is not necessarily the broker's settlement price."}
