"""Check API connectivity and verify the assumptions the app relies on. Uses 1-2 requests."""
import _common  # noqa: F401
import time

from api.client import OTChartsClient
from api.schemas import OTChartsError
from config import SYMBOL, TIMEFRAME_S, VENUE
from data.validation import clean_candles


def main() -> int:
    try:
        c = OTChartsClient()
        u = c.usage()      # free call, never counts against quota
        print("plan:", u.get("planName") or u.get("plan"), "| books:", u.get("books"),
              "| requests:", u.get("requests"), "| streams:", u.get("streams"))
        v = {x["id"]: x for x in c.venues().get("venues", [])}
        print(f"venue '{VENUE}' open on your plan:", v.get(VENUE, {}).get("open"),
              "| timeframes:", v.get(VENUE, {}).get("timeframes"))
        b = c.fetch_candles(VENUE, SYMBOL, TIMEFRAME_S, limit=50)
    except OTChartsError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    clean, issues = clean_candles(b.candles, TIMEFRAME_S)
    now = b.server_time or time.time()
    last = clean[-1]
    print(f"received {len(b.candles)} candles, {len(b.rejected)} rejected, {len(issues)} issues, "
          f"provider gaps={b.gap_runs} runs/{b.gap_missing_bars} missing")
    print("timestamps multiple of 60:", all(x.ts % 60 == 0 for x in clean))
    print(f"newest bar ts={last.ts}; its age = {now - last.ts:.0f}s  "
          f"({'FORMING (age < 60s)' if now - last.ts < 60 else 'complete'})")
    print("clock offset server-local (s):", None if b.clock_offset_s is None else round(b.clock_offset_s, 1))
    print("Check manually: is `time` the bar OPEN? (newest bar age < 60 s while forming implies yes.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
