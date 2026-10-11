"""Walk-forward backtest + optional hypothetical simulator. Results are saved for the dashboard."""
import _common  # noqa: F401
import argparse
import json

from config import DB_PATH, Settings
from data.database import Database
from evaluation.backtest import run_backtest, simulate_from_backtest, summary_for_storage
from evaluation.simulator import SimConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=int, default=500)
    ap.add_argument("--payout", type=float, default=0.85, help="profit ratio on a win, e.g. 0.85")
    ap.add_argument("--delay", type=float, default=0.0, help="execution delay in seconds (0-60)")
    ap.add_argument("--tie-rule", choices=["refund", "loss"], default="refund")
    a = ap.parse_args()
    s, db = Settings(), Database(DB_PATH)
    candles = db.load_candles(s.venue, s.symbol, s.timeframe_s)
    res = run_backtest(candles, s, test_size=a.test_size)
    if not res["sufficient"]:
        print(res["message"])
        db.save_backtest("all", 0, 0, res.get("n_usable_rows", 0), summary_for_storage(res))
        return 1
    print(f"range {res['date_range_utc']}  DOJI outcomes {res['doji_outcomes_in_range']}  "
          f"break-even win rate {res['break_even_win_rate_at_payout']:.3f}")
    for name, rep in res["models"].items():
        x = rep["all_rows"]
        print(f"{name:9s} n={x['n_predictions']:6d} acc={x['accuracy']:.4f} "
              f"CI95=({x['accuracy_ci95'][0]:.3f},{x['accuracy_ci95'][1]:.3f}) bal={x['balanced_accuracy']:.4f} "
              f"p(vs breakeven)={rep['p_value_vs_breakeven']:.3f}\n          -> {res['verdicts'][name]}")
    cfg = SimConfig(payout_ratio=a.payout, delay_s=a.delay, tie_rule=a.tie_rule, tol=s.doji_tolerance)
    for name in ("logreg", "hgb"):
        sim = simulate_from_backtest(res, name, candles, s, cfg)
        print(f"HYPOTHETICAL simulation [{name}] (not real money):", json.dumps(sim, default=str))
    db.save_backtest("all", 0, 0, res["n_usable_rows"], summary_for_storage(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
