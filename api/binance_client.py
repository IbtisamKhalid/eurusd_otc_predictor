"""Binance REST adapter for public market data.

No API keys required. Fetches real 1-minute OHLCV klines from:
https://api.binance.com/api/v3/klines
"""
from __future__ import annotations
import email.utils
import logging
import time
from typing import Optional

import requests

from api.schemas import Candle, CandleBatch, NetworkError, ResponseFormatError, UpstreamError

log = logging.getLogger(__name__)
BINANCE_BASE_URL = "https://api.binance.com"
USER_AGENT = "eurusd-otc-predictor/1.0 (binance-adapter)"

TIMEFRAME_MAP = {
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
}


class BinanceClient:
    def __init__(self, base_url: str = BINANCE_BASE_URL, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout

    def fetch_candles(self, venue: str, symbol: str, timeframe_s: int,
                      limit: int = 500, before: Optional[int] = None) -> CandleBatch:
        """Fetch 1m OHLCV bars.

        `before` is unix seconds; Binance expects `endTime` in milliseconds.
        """
        interval = TIMEFRAME_MAP.get(timeframe_s, "1m")
        # Normalize symbol: e.g. "EUR/USDT" or "EURUSD" -> "EURUSDT"
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        if clean_symbol in ("EURUSD", "EURUSDOTC"):
            clean_symbol = "EURUSDT"

        params: dict = {
            "symbol": clean_symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if before is not None:
            # endTime in ms: end strictly before `before`
            params["endTime"] = int(before * 1000) - 1

        url = f"{self.base_url}/api/v3/klines"
        t0 = time.time()
        try:
            resp = self.session.get(url, params=params, headers={"User-Agent": USER_AGENT},
                                    timeout=self.timeout)
        except requests.RequestException as exc:
            raise NetworkError(f"Binance connection failed: {exc}") from exc

        if resp.status_code >= 500:
            raise UpstreamError(f"Binance 5xx error: {resp.status_code} {resp.text[:100]}")
        if resp.status_code != 200:
            raise ResponseFormatError(f"Binance HTTP {resp.status_code}: {resp.text[:100]}")

        # Server clock parsing from Date header
        server_time, clock_offset = None, None
        date_hdr = resp.headers.get("Date")
        if date_hdr:
            try:
                server_time = email.utils.parsedate_to_datetime(date_hdr).timestamp()
                clock_offset = server_time - (t0 + (time.time() - t0) / 2)
            except Exception:
                pass

        try:
            raw_klines = resp.json()
        except Exception as exc:
            raise ResponseFormatError(f"Non-JSON response from Binance: {exc}") from exc

        if not isinstance(raw_klines, list):
            raise ResponseFormatError(f"Unexpected klines payload: {raw_klines}")

        candles: list[Candle] = []
        rejected: list[tuple[object, str]] = []

        for row in raw_klines:
            try:
                # row[0]: open time ms
                # row[1]: open, row[2]: high, row[3]: low, row[4]: close, row[5]: volume
                ts = int(row[0] // 1000)
                o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                vol = float(row[5]) if len(row) > 5 else None
                candles.append(Candle(ts=ts, open=o, high=h, low=l, close=c, volume=vol))
            except Exception as exc:
                rejected.append((row, str(exc)))

        # Sort chronologically (Binance returns oldest -> newest)
        candles.sort(key=lambda x: x.ts)

        return CandleBatch(
            candles=candles,
            rejected=rejected,
            interval=interval,
            symbol=clean_symbol,
            server_time=server_time,
            clock_offset_s=clock_offset,
            raw={"count": len(candles)}
        )
