"""Central configuration. Secrets come from the environment only."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("OTC_DB_PATH", BASE_DIR / "data_storage" / "predictor.db"))

API_BASE_URL = "https://otcharts.com"
VENUE = os.environ.get("OTC_VENUE", "binance")        # 'binance' (free, live 24/7) | 'iq' (OTCharts)
SYMBOL = os.environ.get("OTC_SYMBOL", "EURUSDT")      # 'EURUSDT' (binance) | 'EURUSD-OTC' (iq)
TIMEFRAME_S = 60
MIN_POLL_INTERVAL_S = 30


def api_key() -> str | None:
    key = os.environ.get("OTCHARTS_API_KEY")
    return key.strip() if key and key.strip() else None


@dataclass
class Settings:
    venue: str = VENUE
    symbol: str = SYMBOL
    timeframe_s: int = TIMEFRAME_S
    poll_interval_s: int = 60
    max_staleness_s: int = 150       # data older than this => NO TRADE (cannot be disabled)
    model_name: str = "logreg"       # majority | momentum | logreg | hgb
    decision_threshold: float = 0.55 # min prob of predicted side to emit a signal
    min_train_samples: int = 2000
    retention_days: int = 90
    display_candles: int = 120
    doji_tolerance: float = 1e-5     # |close-open| <= tol => DOJI
    payout_ratio: float = 0.85       # only used by the offline simulator

    def __post_init__(self) -> None:
        # Settings may never bypass safety checks.
        self.poll_interval_s = max(self.poll_interval_s, MIN_POLL_INTERVAL_S)
        self.max_staleness_s = min(max(self.max_staleness_s, 2 * self.timeframe_s // 2), 600)
        self.decision_threshold = min(max(self.decision_threshold, 0.5), 0.95)
        self.min_train_samples = max(self.min_train_samples, 300)
