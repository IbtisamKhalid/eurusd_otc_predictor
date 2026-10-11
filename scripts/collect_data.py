"""Collect and validate historical candles into SQLite.
Paging older than the live window needs a paid OTCharts plan (Build+); on the free tier only the
newest window is returned and the script says so."""
import _common  # noqa: F401
import argparse

from api.client import OTChartsClient
from api.schemas import OTChartsError
from config import DB_PATH, Settings
from data.collector import Collector
from data.database import Database


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=5000, help="target number of 1m bars (default 5000)")
    a = ap.parse_args()
    s, db = Settings(), Database(DB_PATH)
    try:
        col = Collector(OTChartsClient(), db, s)
    except OTChartsError as exc:
        print(exc)
        return 1
    res = col.backfill(target_bars=a.bars)
    df = db.load_candles(s.venue, s.symbol, s.timeframe_s)
    print(res)
    print(f"stored completed candles: {len(df)}", "" if df.empty else f"({df['dt'].min()} -> {df['dt'].max()})")
    print(db.recent_events(10).to_string(index=False))
    if not res["ok"]:
        print("\nBackfill stopped early: see error above. 'PlanError' on a later page = your plan does not "
              "include `before=` history. Run the dashboard or this script repeatedly over days to accumulate.")
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
