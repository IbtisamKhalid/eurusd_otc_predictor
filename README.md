# EUR/USD OTC Candle Predictor (read-only research tool)

Predicts whether the **next 1-minute IQ Option EURUSD-OTC candle closes UP or DOWN** (or returns
**NO TRADE**), using data from the OTCharts API. It **never places trades**, makes **no accuracy
guarantee**, and treats "no demonstrable edge" as a normal, expected result.

> OTC series are broker-generated prices. OTCharts itself reports that the OTC instruments it measured
> did not track the real markets they are named after. Ordinary EUR/USD indicator behaviour may not apply.

## What is verified vs. not

| Item | Status |
|---|---|
| OTCharts contract (bearer auth, `GET /v1/candles` params, response fields, `gaps`, error codes, `iq` venue = `EURUSD-OTC`, UTC timestamps) | Read from https://otcharts.com/api on 2026-10-10 and implemented exactly |
| All code paths: parsing, retries/backoff, validation, features, leakage checks, calibration, walk-forward, simulator, outcome matching, DB | **Unit-tested (55 tests, synthetic fixtures only)** |
| Live HTTP calls to OTCharts | **NOT tested** (no key/network in the build environment). Run `python scripts\verify_api.py` |
| Streamlit dashboard | Syntax-checked only; **not launched** in the build environment. Expect to fix small UI issues on first run |
| Assumption: candle `time` = bar open, newest bar may still be forming | Not stated in docs. App is conservative either way (a bar counts as complete only after `time + 60 s`); `verify_api.py` prints evidence |
| Not implemented | `/v1/stream` (SSE) live ticks; REST polling is used instead (see cost note below) |

Docs conflict to resolve on your account page: the API reference says the free tier is **200 requests in
the first month, no streams**; the PyPI page says **1,500/week + one stream**. Polling once a minute costs
1,440 requests/day, so it needs a paid plan (Lite = 7,000/day). `GET /v1/usage` is free and shows your real limits.

## Windows setup (PowerShell)

```powershell
# 1. Install Python 3.11 from python.org (tick "Add to PATH" and the py launcher)
cd path\to\eurusd_otc_predictor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
$env:OTCHARTS_API_KEY = "otc_live_REPLACE_ME"   # this session only; or put it in .env (never commit it)
python scripts\verify_api.py          # connectivity + assumption checks
python scripts\collect_data.py --bars 5000
python scripts\train_model.py
python scripts\run_backtest.py
streamlit run app.py                  # dashboard at http://localhost:8501
python -m pytest -q                   # tests
```
Or double-click `setup_windows.bat` then `run_dashboard.bat`.

## Collecting enough data
* The free tier / any plan without `before=` returns only the live window (up to 5,000 bars). Paging further
  back needs Build ($49) or higher. Otherwise run the dashboard or `collect_data.py` repeatedly for days.
* Minimums enforced: 2,000 usable rows to train (configurable, floor 300), 400 validation rows to calibrate,
  and `min_train + 500` rows for a backtest. Below that the app says "insufficient data".
* Rule of thumb: distinguishing 53% from 50% at 95% confidence needs roughly 4,000+ decided predictions,
  and beating a ~54% break-even (85% payout) needs far more. The dashboard shows 95% intervals and p-values.

## How it avoids fooling you
* Only **completed** candles feed features; the forming candle is drawn grey and excluded.
* Row *t* uses bars <= *t*; its label is bar *t+1*. Tests prove future bars cannot change past features.
* Gaps invalidate the rows whose 30-bar lookback contains them; provider `gaps` are surfaced.
* Walk-forward expanding windows with a 15-row embargo; scaler and calibrator fitted inside each fold.
* Probabilities are shown **only** if cross-fitted validation ECE is within 0.02 of the sampling-noise floor;
  otherwise the model returns NO TRADE and its raw call is logged as a *shadow* prediction for forward testing.
* DOJI (|close-open| <= tolerance) is excluded from fitting and counted separately; never scored as a win.
* Stale (> limit, hard cap 600 s), disconnected, or clock-skewed data => NO TRADE. Settings cannot bypass this
  (polling floor 30 s, threshold floor 0.50, training floor 300 rows).
* Simulator results are hypothetical; the broker's real entry/settlement price is unknown.

## Layout
`app.py` dashboard - `config.py` - `api/` adapter - `data/` collector, validation, SQLite - `features/` -
`models/` baselines, calibration, training, prediction - `evaluation/` metrics, walk-forward, backtest,
simulator, forward-test matching - `dashboard/` charts, panels - `scripts/` CLI - `tests/`.

## Known limitations / next steps
* Signal timing: a candle is known to be complete only after it closes, so each prediction is computed some
  seconds *after* the target candle opened (shown as lag). It is for research, not execution.
* Implement `/v1/stream` to cut request cost and latency; validate the bar-open assumption; consider the
  `forex` book as a comparison series; add regime-specific models only if forward tests justify them.
* Expect majority/momentum/logreg/HGB to be near 50% on efficient or synthetic feeds. Treat any apparent edge
  as unproven until it survives thousands of forward-test predictions.
