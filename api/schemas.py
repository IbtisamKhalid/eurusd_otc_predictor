"""Typed containers and exceptions for the OTCharts adapter."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


class OTChartsError(Exception):
    """Base class. Messages never contain credentials."""


class ConfigError(OTChartsError):
    """OTCHARTS_API_KEY missing."""


class AuthError(OTChartsError):          # HTTP 401
    pass


class PlanError(OTChartsError):          # HTTP 402 / 403
    pass


class QuotaExceeded(OTChartsError):      # HTTP 429
    pass


class BadRequest(OTChartsError):         # HTTP 400 / 404
    pass


class UpstreamError(OTChartsError):      # HTTP 5xx after retries
    pass


class NetworkError(OTChartsError):       # timeouts / connection failures after retries
    pass


class ResponseFormatError(OTChartsError):
    pass


@dataclass(frozen=True)
class Candle:
    ts: int            # unix seconds, UTC (IQ book is true UTC per OTCharts docs)
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


@dataclass
class CandleBatch:
    candles: list[Candle]
    rejected: list[tuple[object, str]] = field(default_factory=list)  # (raw item, reason)
    gap_runs: int = 0                    # provider-reported 'gaps.runs'
    gap_missing_bars: int = 0            # provider-reported 'gaps.missingBars'
    exhausted: Optional[bool] = None
    interval: Optional[str] = None
    symbol: Optional[str] = None
    server_time: Optional[float] = None  # from HTTP Date header (1 s resolution)
    clock_offset_s: Optional[float] = None  # server_time - local_time (approx.)
    raw: Optional[dict] = None
