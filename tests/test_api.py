import logging
import pytest
from api.client import OTChartsClient, parse_candles_payload
from api.schemas import (AuthError, BadRequest, ConfigError, NetworkError, PlanError, QuotaExceeded,
                         ResponseFormatError, UpstreamError)
import requests

SECRET = "otc_live_SUPERSECRET123"
GOOD = {"candles": [{"open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "volume": 3.0, "time": 1788103260},
                    {"open": 1.15, "high": 1.16, "low": 1.14, "close": 1.15, "time": 1788103320}],
        "count": 2, "interval": "60s", "symbol": "EURUSD-OTC", "gaps": {"runs": 1, "missingBars": 4}}


class Resp:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = text if text is not None else str(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(script, **kw):
    sleeps = []
    sess = FakeSession(script)
    c = OTChartsClient(key=SECRET, session=sess, sleep=sleeps.append, backoff_base=1.0, **kw)
    return c, sess, sleeps


def test_missing_key_raises(monkeypatch=None):
    import os
    old = os.environ.pop("OTCHARTS_API_KEY", None)
    try:
        with pytest.raises(ConfigError):
            OTChartsClient(key=None)
    finally:
        if old is not None:
            os.environ["OTCHARTS_API_KEY"] = old


def test_fetch_success_and_params():
    c, sess, _ = client([Resp(200, GOOD, {"Date": "Tue, 15 Nov 1994 08:12:31 GMT"})])
    b = c.fetch_candles("iq", "EURUSD-OTC", 60, 2)
    assert len(b.candles) == 2 and b.candles[1].volume is None
    assert b.gap_runs == 1 and b.gap_missing_bars == 4
    url, params, headers = sess.calls[0]
    assert url == "https://otcharts.com/v1/candles"
    assert params == {"venue": "iq", "symbol": "EURUSD-OTC", "tf": 60, "limit": 2}
    assert headers["Authorization"] == f"Bearer {SECRET}"
    assert b.server_time == 784887151.0 and b.clock_offset_s is not None


def test_before_param_and_limit_clamp():
    c, sess, _ = client([Resp(200, GOOD)])
    c.fetch_candles("iq", "EURUSD-OTC", 60, 99999, before=123)
    assert sess.calls[0][1]["limit"] == 5000 and sess.calls[0][1]["before"] == 123


def test_error_mapping_no_retry():
    for status, exc in [(401, AuthError), (402, PlanError), (403, PlanError), (429, QuotaExceeded),
                        (400, BadRequest), (404, BadRequest)]:
        c, sess, sleeps = client([Resp(status, {"error": "nope"})])
        with pytest.raises(exc):
            c.fetch_candles("iq", "X")
        assert len(sess.calls) == 1 and sleeps == []


def test_cdn_1010_hint():
    c, _, _ = client([Resp(403, None, text="error code: 1010")])
    with pytest.raises(PlanError) as e:
        c.fetch_candles("iq", "X")
    assert "1010" in str(e.value)


def test_retry_with_exponential_backoff_then_success():
    c, sess, sleeps = client([Resp(502, {"error": "venue down"}), Resp(503, {"error": "busy"}), Resp(200, GOOD)])
    assert len(c.fetch_candles("iq", "EURUSD-OTC").candles) == 2
    assert len(sess.calls) == 3 and len(sleeps) == 2
    assert 1.0 <= sleeps[0] < 1.5 and 2.0 <= sleeps[1] < 2.5


def test_retry_after_honoured():
    c, _, sleeps = client([Resp(503, {"error": "busy"}, {"Retry-After": "7"}), Resp(200, GOOD)])
    c.fetch_candles("iq", "X")
    assert 7.0 <= sleeps[0] < 7.5


def test_upstream_error_after_exhausted_retries():
    c, sess, _ = client([Resp(502, {"error": "x"})] * 3, max_retries=2)
    with pytest.raises(UpstreamError):
        c.fetch_candles("iq", "X")
    assert len(sess.calls) == 3


def test_network_failure_raises_network_error():
    c, sess, sleeps = client([requests.ConnectionError(f"boom {SECRET}")] * 3, max_retries=2)
    with pytest.raises(NetworkError) as e:
        c.fetch_candles("iq", "X")
    assert SECRET not in str(e.value) and len(sleeps) == 2


def test_invalid_json_and_shape():
    c, _, _ = client([Resp(200, None, text="<html>")])
    with pytest.raises(ResponseFormatError):
        c.fetch_candles("iq", "X")
    with pytest.raises(ResponseFormatError):
        parse_candles_payload({"nope": []})


def test_bad_candles_are_rejected_not_repaired():
    payload = {"candles": [{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "time": 60},
                           {"open": "x", "high": 2, "low": 0.5, "close": 1.5, "time": 120},
                           {"open": 1, "high": 2, "low": 0.5, "time": 180},
                           "garbage", {"open": True, "high": 2, "low": 1, "close": 1, "time": 240},
                           {"open": float("nan"), "high": 2, "low": 1, "close": 1, "time": 300}]}
    b = parse_candles_payload(payload)
    assert len(b.candles) == 1 and len(b.rejected) == 5


def test_secret_never_in_logs_or_errors(caplog=None):
    handler_records = []

    class H(logging.Handler):
        def emit(self, r):
            handler_records.append(r.getMessage())
    h = H()
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        c, _, _ = client([Resp(502, {"error": "x"}), Resp(401, {"error": f"key {SECRET} bad"})])
        try:
            c.fetch_candles("iq", "X")
        except AuthError as e:
            handler_records.append(str(e))
    finally:
        logging.getLogger().removeHandler(h)
    assert handler_records and all(SECRET not in m for m in handler_records)
