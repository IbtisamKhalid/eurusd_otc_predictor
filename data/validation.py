"""Candle validation, de-duplication, ordering and gap detection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

from api.schemas import Candle


@dataclass(frozen=True)
class Issue:
    kind: str      # invalid_ohlc | duplicate | duplicate_conflict | out_of_order | misaligned | gap
    ts: Optional[int]
    detail: str


def ohlc_problem(c: Candle) -> Optional[str]:
    if min(c.open, c.high, c.low, c.close) <= 0:
        return "non-positive price"
    if c.high < max(c.open, c.close, c.low):
        return "high below open/close/low"
    if c.low > min(c.open, c.close, c.high):
        return "low above open/close/high"
    if c.volume is not None and c.volume < 0:
        return "negative volume"
    return None


def clean_candles(candles: Sequence[Candle], tf: int = 60) -> tuple[list[Candle], list[Issue]]:
    """Return strictly increasing, unique, aligned, internally consistent candles + issues."""
    issues: list[Issue] = []
    prev_ts: Optional[int] = None
    for c in candles:
        if prev_ts is not None and c.ts < prev_ts:
            issues.append(Issue("out_of_order", c.ts, f"{c.ts} arrived after {prev_ts}"))
        prev_ts = c.ts if prev_ts is None else max(prev_ts, c.ts)
    by_ts: dict[int, Candle] = {}
    for c in candles:
        bad = ohlc_problem(c)
        if bad:
            issues.append(Issue("invalid_ohlc", c.ts, bad))
            continue
        if c.ts % tf != 0:
            issues.append(Issue("misaligned", c.ts, f"timestamp not a multiple of {tf}s"))
            continue
        if c.ts in by_ts:
            same = by_ts[c.ts] == c
            issues.append(Issue("duplicate" if same else "duplicate_conflict", c.ts,
                                "identical repeat" if same else "differing values; kept last"))
        by_ts[c.ts] = c
    clean = [by_ts[k] for k in sorted(by_ts)]
    for a, b in zip(clean, clean[1:]):
        if b.ts - a.ts > tf:
            issues.append(Issue("gap", a.ts, f"{(b.ts - a.ts) // tf - 1} bar(s) missing after {a.ts}"))
    return clean, issues


def lag_seconds(last_complete_ts: int, tf: int, now_server: float) -> float:
    """Seconds since the last completed bar closed (>= 0 when fresh and clocks agree)."""
    return now_server - (last_complete_ts + tf)


def is_stale(last_complete_ts: Optional[int], tf: int, now_server: float,
             max_staleness_s: float) -> bool:
    if last_complete_ts is None:
        return True
    return lag_seconds(last_complete_ts, tf, now_server) > max_staleness_s
