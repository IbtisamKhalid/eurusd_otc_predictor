"""Collector pipeline with a scripted fake client (no network, synthetic data, in-memory DB)."""
import pytest
from api.schemas import Candle, CandleBatch, NetworkError
from config import Settings
from data.collector import Collector
from data.database import Database

T0 = 1_700_000_040 - 1_700_000_040 % 60


class FakeClient:
    def __init__(self, batches):
        self.batches = list(batches)

    def fetch_candles(self, venue, symbol, tf, limit, before=None):
        b = self.batches.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def cs(*idx, c=1.05):
    return [Candle(T0 + 60 * i, 1.0, 1.1, 0.9, c) for i in idx]


def test_forming_bar_not_stored_and_disconnect_is_reported():
    db, s = Database(":memory:"), Settings()
    now = T0 + 60 * 3 + 20                       # bar 3 is 20 s old -> forming
    batch = CandleBatch(candles=cs(0, 1, 2, 3), gap_runs=0, server_time=now, clock_offset_s=0.0)
    col = Collector(FakeClient([batch, NetworkError("down")]), db, s, clock=lambda: now)
    r = col.fetch_and_store()
    assert r["ok"] and r["inserted"] == 3 and [c.ts for c in col.state.forming] == [T0 + 180]
    assert len(db.load_candles(s.venue, s.symbol, 60)) == 3
    r = col.fetch_and_store()
    assert not r["ok"] and not col.state.connected and "NetworkError" in col.state.last_error
    assert len(db.load_candles(s.venue, s.symbol, 60)) == 3          # nothing invented on failure
    assert "api_error" in set(db.recent_events()["kind"])


def test_issues_logged_and_duplicates_not_double_stored():
    db, s = Database(":memory:"), Settings()
    now = T0 + 60 * 10
    bad = Candle(T0 + 60 * 4, 1.0, 0.5, 0.9, 1.0)                    # invalid OHLC
    batch = CandleBatch(candles=cs(0, 1, 1, 2) + [bad] + cs(6), rejected=[({"x": 1}, "KeyError")],
                        gap_runs=1, gap_missing_bars=3, clock_offset_s=0.0)
    col = Collector(FakeClient([batch, CandleBatch(candles=cs(0, 1, 2))]), db, s, clock=lambda: now)
    r = col.fetch_and_store()
    assert r["inserted"] == 4 and r["gap_runs"] == 1
    kinds = set(db.recent_events(50)["kind"])
    assert {"duplicate", "invalid_ohlc", "gap", "rejected_candle"} <= kinds
    assert col.fetch_and_store()["inserted"] == 0                    # idempotent re-delivery


def test_revision_is_logged_but_original_kept():
    db, s = Database(":memory:"), Settings()
    now = T0 + 600
    col = Collector(FakeClient([CandleBatch(candles=cs(0, 1)), CandleBatch(candles=cs(1, c=1.09))]),
                    db, s, clock=lambda: now)
    col.fetch_and_store()
    r = col.fetch_and_store()
    assert r["conflicts"] == 1 and db.load_candles(s.venue, s.symbol, 60).loc[1, "close"] == 1.05
