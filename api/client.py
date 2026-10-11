"""OTCharts REST adapter.

Verified against https://otcharts.com/api (docs fetched 2026-10-10):
  * Auth: ``Authorization: Bearer <key>``; base URL https://otcharts.com (HTTPS only)
  * GET /v1/candles?venue=iq&symbol=EURUSD-OTC&tf=60&limit=N[&before=unix]
  * Response: {"candles":[{open,high,low,close,volume,time}], "count", "interval",
               "symbol", "gaps":{"runs","missingBars"}, "exhausted"?}
  * Errors: JSON {"error": ...}; 401/402/403/429/400/404/502. A 403 with a non-JSON
    body "error code: 1010" is a CDN block of Python's urllib User-Agent (we use
    ``requests`` and also set our own User-Agent).
  * The docs warn: do NOT poll /v1/candles on a timer; /v1/stream (SSE) is the intended
    live path (paid plans). This adapter implements REST only.

NOT documented (so not assumed): whether the newest bar is still forming. Callers must
treat the newest bar as incomplete unless now >= bar.ts + timeframe.
"""
from __future__ import annotations

import email.utils
import logging
import random
import time
from typing import Callable, Optional

import requests

from config import API_BASE_URL, api_key
from api.schemas import (AuthError, BadRequest, Candle, CandleBatch, ConfigError, NetworkError,
                         PlanError, QuotaExceeded, ResponseFormatError, UpstreamError)

log = logging.getLogger(__name__)
USER_AGENT = "eurusd-otc-predictor/1.0 (research; read-only)"
_RETRY_STATUS = {500, 502, 503, 504}


class OTChartsClient:
    def __init__(self, key: Optional[str] = None, base_url: str = API_BASE_URL,
                 session: Optional[requests.Session] = None, timeout: float = 15.0,
                 max_retries: int = 4, backoff_base: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time) -> None:
        self._key = key if key is not None else api_key()
        if not self._key:
            raise ConfigError("OTCHARTS_API_KEY is not set. Create a key at "
                              "https://otcharts.com/account and set it in your environment.")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout, self.max_retries, self.backoff_base = timeout, max_retries, backoff_base
        self._sleep, self._clock = sleep, clock

    # -- internals ---------------------------------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT,
                "Accept": "application/json"}

    @staticmethod
    def _error_text(resp: requests.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict) and "error" in body:
                return str(body["error"])[:300]
        except ValueError:
            pass
        return (resp.text or "")[:120].replace("\n", " ")

    def _backoff(self, attempt: int, retry_after: Optional[float] = None) -> None:
        delay = retry_after if retry_after is not None else self.backoff_base * (2 ** attempt)
        delay = min(delay, 60.0) + random.uniform(0, 0.25 * self.backoff_base)
        self._sleep(delay)

    def _get(self, path: str, params: dict) -> tuple[dict, requests.Response, float]:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = self._clock()
            try:
                resp = self.session.get(url, params=params, headers=self._headers(),
                                        timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                log.warning("network error on %s (attempt %d): %s", path, attempt + 1,
                            type(exc).__name__)
                if attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                raise NetworkError(f"Could not reach OTCharts after {attempt + 1} attempts "
                                   f"({type(exc).__name__}). Check your internet connection.") from None
            t1 = self._clock()
            status = resp.status_code
            if status == 200:
                try:
                    return resp.json(), resp, (t0 + t1) / 2
                except ValueError:
                    raise ResponseFormatError("200 response was not valid JSON") from None
            msg = self._error_text(resp)
            if status == 401:
                raise AuthError("401: API key missing, revoked, unknown, or account email "
                                "unconfirmed. (A password reset revokes all keys.)")
            if status in (402, 403):
                hint = " (CDN block 1010: client User-Agent refused)" if "1010" in msg else ""
                raise PlanError(f"{status}: plan does not open this book/instrument{hint}: {msg}")
            if status == 429:
                raise QuotaExceeded(f"429: quota exhausted or streams in use: {msg}")
            if status in (400, 404):
                raise BadRequest(f"{status}: {msg}")
            if status in _RETRY_STATUS and attempt < self.max_retries:
                ra = resp.headers.get("Retry-After")
                try:
                    ra_f = float(ra) if ra is not None else None
                except ValueError:
                    ra_f = None
                log.warning("upstream %d on %s (attempt %d)", status, path, attempt + 1)
                self._backoff(attempt, ra_f)
                continue
            raise UpstreamError(f"HTTP {status}: {msg}")
        raise NetworkError(str(last_exc))  # pragma: no cover

    # -- public API --------------------------------------------------------------
    def fetch_candles(self, venue: str, symbol: str, tf: int = 60, limit: int = 300,
                      before: Optional[int] = None) -> CandleBatch:
        params: dict = {"venue": venue, "symbol": symbol, "tf": tf,
                        "limit": max(1, min(int(limit), 5000))}
        if before is not None:
            params["before"] = int(before)
        payload, resp, local_mid = self._get("/v1/candles", params)
        return parse_candles_payload(payload, resp.headers.get("Date"), local_mid)

    def usage(self) -> dict:
        """Free call: does not count against quota."""
        payload, _, _ = self._get("/v1/usage", {})
        return payload

    def venues(self) -> dict:
        payload, _, _ = self._get("/v1/venues", {})
        return payload


def _num(v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"not numeric: {v!r}")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError("non-finite")
    return f


def parse_candles_payload(payload: object, date_header: Optional[str] = None,
                          local_mid: Optional[float] = None) -> CandleBatch:
    """Validate the JSON shape. Bad individual candles are rejected with a reason,
    never silently repaired."""
    if not isinstance(payload, dict) or not isinstance(payload.get("candles"), list):
        raise ResponseFormatError("Response missing 'candles' list")
    out: list[Candle] = []
    rejected: list[tuple[object, str]] = []
    for item in payload["candles"]:
        try:
            if not isinstance(item, dict):
                raise ValueError("candle is not an object")
            vol = item.get("volume")
            out.append(Candle(ts=int(_num(item["time"])), open=_num(item["open"]),
                              high=_num(item["high"]), low=_num(item["low"]),
                              close=_num(item["close"]),
                              volume=_num(vol) if vol is not None else None))
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append((item, f"{type(exc).__name__}: {exc}"))
    gaps = payload.get("gaps") if isinstance(payload.get("gaps"), dict) else {}
    server_time = offset = None
    if date_header:
        try:
            server_time = email.utils.parsedate_to_datetime(date_header).timestamp()
            if local_mid is not None:
                offset = server_time - local_mid
        except (TypeError, ValueError):
            pass
    return CandleBatch(candles=out, rejected=rejected,
                       gap_runs=int(gaps.get("runs", 0) or 0),
                       gap_missing_bars=int(gaps.get("missingBars", 0) or 0),
                       exhausted=payload.get("exhausted"), interval=payload.get("interval"),
                       symbol=payload.get("symbol"), server_time=server_time,
                       clock_offset_s=offset, raw=payload)
