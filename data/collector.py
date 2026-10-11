"""Fetch -> validate -> store. Never fabricates data: on failure it records the error and
leaves the last-good state untouched."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from api.client import OTChartsClient
from api.schemas import Candle, OTChartsError
from config import Settings
from data.candles import split_complete
from data.database import Database
from data.validation import clean_candles

log = logging.getLogger(__name__)


@dataclass
class CollectorState:
    last_success_local: Optional[float] = None   # wall-clock of last successful fetch
    last_error: Optional[str] = None
    clock_offset_s: float = 0.0                  # server - local (from HTTP Date header), best effort
    forming: list[Candle] = field(default_factory=list)
    last_gap_runs: int = 0
    last_gap_missing: int = 0
    connected: bool = False

    def server_now(self, local_now: Optional[float] = None) -> float:
        return (local_now if local_now is not None else time.time()) + self.clock_offset_s


class Collector:
    def __init__(self, client: OTChartsClient, db: Database, settings: Settings,
                 clock: Callable[[], float] = time.time) -> None:
        self.client, self.db, self.s, self._clock = client, db, settings, clock
        self.state = CollectorState()

    def fetch_and_store(self, limit: int = 300, before: Optional[int] = None) -> dict:
        s = self.s
        try:
            batch = self.client.fetch_candles(s.venue, s.symbol, s.timeframe_s, limit, before)
        except OTChartsError as exc:
            self.state.connected = False
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            self.db.log_event("api_error", self.state.last_error)
            return {"ok": False, "error": self.state.last_error, "inserted": 0}

        if batch.clock_offset_s is not None:
            self.state.clock_offset_s = batch.clock_offset_s
        for raw, why in batch.rejected:
            self.db.log_event("rejected_candle", f"{why}: {str(raw)[:150]}")
        if batch.raw is not None:
            self.db.save_raw(s.venue, s.symbol, s.timeframe_s, batch.raw)

        clean, issues = clean_candles(batch.candles, s.timeframe_s)
        for i in issues:
            self.db.log_event(i.kind, i.detail, i.ts)
        done, forming = split_complete(clean, self.state.server_now(self._clock()), s.timeframe_s)
        inserted, conflicts = self.db.insert_candles(s.venue, s.symbol, s.timeframe_s, done)
        for ts in conflicts:
            self.db.log_event("revision_ignored", "provider re-delivered differing values", ts)

        self.state.forming = forming
        self.state.last_gap_runs, self.state.last_gap_missing = batch.gap_runs, batch.gap_missing_bars
        self.state.last_success_local = self._clock()
        self.state.last_error = None
        self.state.connected = True
        return {"ok": True, "inserted": inserted, "received": len(batch.candles),
                "rejected": len(batch.rejected), "gap_runs": batch.gap_runs,
                "gap_missing_bars": batch.gap_missing_bars, "conflicts": len(conflicts)}

    def backfill(self, target_bars: int = 5000, page: int = 1500, max_pages: int = 40) -> dict:
        """Page backwards with `before=` (paid plans, Build+). On a free plan the first page
        succeeds and later pages raise PlanError, which is reported, not hidden."""
        total, before, pages = 0, None, 0
        while total < target_bars and pages < max_pages:
            res = self.fetch_and_store(limit=min(page, target_bars - total), before=before)
            pages += 1
            if not res["ok"]:
                return {"ok": False, "pages": pages, "inserted": total, "error": res["error"]}
            total += res["inserted"]
            df = self.db.load_candles(self.s.venue, self.s.symbol, self.s.timeframe_s, limit=None)
            oldest = int(df["ts"].min()) if len(df) else None
            if res["received"] == 0 or oldest is None or oldest == before:
                break
            before = oldest
        return {"ok": True, "pages": pages, "inserted": total}
