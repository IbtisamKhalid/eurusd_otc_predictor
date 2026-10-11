"""SQLite persistence. All timestamps are UTC unix seconds."""
from __future__ import annotations
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from api.schemas import Candle

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_responses(
  id INTEGER PRIMARY KEY, fetched_at REAL NOT NULL, venue TEXT, symbol TEXT, tf INTEGER,
  payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candles(
  venue TEXT NOT NULL, symbol TEXT NOT NULL, tf INTEGER NOT NULL, ts INTEGER NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL,
  inserted_at REAL NOT NULL, PRIMARY KEY(venue, symbol, tf, ts));
CREATE TABLE IF NOT EXISTS model_versions(
  version TEXT PRIMARY KEY, name TEXT NOT NULL, trained_at REAL NOT NULL, n_train INTEGER,
  calibration_status TEXT, info TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY, venue TEXT NOT NULL, symbol TEXT NOT NULL, tf INTEGER NOT NULL,
  target_ts INTEGER NOT NULL, signal_ts REAL NOT NULL, based_on_ts INTEGER NOT NULL,
  model_name TEXT NOT NULL, model_version TEXT, direction TEXT NOT NULL,
  raw_direction TEXT, p_up REAL, p_down REAL, calibrated INTEGER NOT NULL DEFAULT 0,
  lag_s REAL, reason TEXT, UNIQUE(venue, symbol, tf, target_ts, model_name));
CREATE TABLE IF NOT EXISTS outcomes(
  prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id), outcome TEXT NOT NULL,
  target_open REAL, target_close REAL, raw_correct INTEGER, signal_correct INTEGER,
  resolved_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS backtests(
  id INTEGER PRIMARY KEY, created_at REAL NOT NULL, model_name TEXT, start_ts INTEGER,
  end_ts INTEGER, n INTEGER, summary TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, at REAL NOT NULL, kind TEXT NOT NULL, ts INTEGER, detail TEXT);
"""


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:") if self.path == ":memory:" else None
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = self._mem or sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            if self._mem is None:
                c.close()

    # -- candles -----------------------------------------------------------------
    def insert_candles(self, venue: str, symbol: str, tf: int, candles: Iterable[Candle]
                       ) -> tuple[int, list[int]]:
        """Insert COMPLETED candles only. Existing bars are never overwritten; a differing
        re-delivery is reported (returned ts list) so the caller can log a data-quality event."""
        inserted, conflicts = 0, []
        now = time.time()
        with self.conn() as c:
            for k in candles:
                row = c.execute("SELECT open,high,low,close FROM candles WHERE venue=? AND symbol=? "
                                "AND tf=? AND ts=?", (venue, symbol, tf, k.ts)).fetchone()
                if row is None:
                    c.execute("INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (venue, symbol, tf, k.ts, k.open, k.high, k.low, k.close, k.volume, now))
                    inserted += 1
                elif tuple(row) != (k.open, k.high, k.low, k.close):
                    conflicts.append(k.ts)
        return inserted, conflicts

    def load_candles(self, venue: str, symbol: str, tf: int, limit: Optional[int] = None
                     ) -> pd.DataFrame:
        q = ("SELECT ts,open,high,low,close,volume FROM candles WHERE venue=? AND symbol=? AND tf=? "
             "ORDER BY ts DESC" + (f" LIMIT {int(limit)}" if limit else ""))
        with self.conn() as c:
            rows = c.execute(q, (venue, symbol, tf)).fetchall()
        df = pd.DataFrame([tuple(r) for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.sort_values("ts").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df

    def save_raw(self, venue: str, symbol: str, tf: int, payload: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO raw_responses(fetched_at,venue,symbol,tf,payload) VALUES(?,?,?,?,?)",
                      (time.time(), venue, symbol, tf, json.dumps(payload)))

    def log_event(self, kind: str, detail: str, ts: Optional[int] = None) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO events(at,kind,ts,detail) VALUES(?,?,?,?)",
                      (time.time(), kind, ts, detail[:500]))

    def recent_events(self, n: int = 50) -> pd.DataFrame:
        with self.conn() as c:
            rows = c.execute("SELECT at,kind,ts,detail FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return pd.DataFrame([tuple(r) for r in rows], columns=["at", "kind", "ts", "detail"])

    # -- models / predictions ----------------------------------------------------
    def save_model_version(self, version: str, name: str, n_train: int, status: str, info: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO model_versions VALUES(?,?,?,?,?,?)",
                      (version, name, time.time(), n_train, status, json.dumps(info, default=str)))

    def insert_prediction(self, **p) -> Optional[int]:
        """Returns new id, or None if a prediction for that (target, model) already exists."""
        cols = ["venue", "symbol", "tf", "target_ts", "signal_ts", "based_on_ts", "model_name",
                "model_version", "direction", "raw_direction", "p_up", "p_down", "calibrated",
                "lag_s", "reason"]
        with self.conn() as c:
            cur = c.execute(f"INSERT OR IGNORE INTO predictions({','.join(cols)}) "
                            f"VALUES({','.join('?' * len(cols))})", [p.get(k) for k in cols])
            return cur.lastrowid if cur.rowcount else None

    def unresolved_predictions(self) -> list[sqlite3.Row]:
        with self.conn() as c:
            return c.execute("SELECT * FROM predictions WHERE id NOT IN (SELECT prediction_id FROM outcomes)"
                             ).fetchall()

    def insert_outcome(self, prediction_id: int, outcome: str, t_open: float, t_close: float,
                       raw_correct: Optional[int], signal_correct: Optional[int]) -> bool:
        with self.conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO outcomes VALUES(?,?,?,?,?,?,?)",
                            (prediction_id, outcome, t_open, t_close, raw_correct, signal_correct, time.time()))
            return bool(cur.rowcount)

    def predictions_with_outcomes(self) -> pd.DataFrame:
        with self.conn() as c:
            rows = c.execute("SELECT p.*, o.outcome, o.raw_correct, o.signal_correct, o.target_open, "
                             "o.target_close FROM predictions p LEFT JOIN outcomes o ON o.prediction_id=p.id "
                             "ORDER BY p.target_ts").fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def save_backtest(self, model_name: str, start_ts: int, end_ts: int, n: int, summary: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO backtests(created_at,model_name,start_ts,end_ts,n,summary) VALUES(?,?,?,?,?,?)",
                      (time.time(), model_name, start_ts, end_ts, n, json.dumps(summary, default=str)))

    def latest_backtests(self) -> pd.DataFrame:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM backtests ORDER BY id DESC LIMIT 20").fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def purge_older_than(self, days: int, venue: str, symbol: str, tf: int) -> int:
        cutoff = int(time.time() - days * 86400)
        with self.conn() as c:
            n = c.execute("DELETE FROM candles WHERE venue=? AND symbol=? AND tf=? AND ts<?",
                          (venue, symbol, tf, cutoff)).rowcount
            c.execute("DELETE FROM raw_responses WHERE fetched_at<?", (time.time() - 7 * 86400,))
            return n

    def export_csv(self, table: str, path: Path | str) -> int:
        if table not in {"candles", "predictions", "outcomes", "events"}:
            raise ValueError("unsupported table")
        with self.conn() as c:
            df = pd.read_sql_query(f"SELECT * FROM {table}", c)
        df.to_csv(path, index=False)
        return len(df)
