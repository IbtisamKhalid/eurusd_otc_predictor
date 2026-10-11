# EUR/USD OTC Candle Predictor — Project Overview & Run Guide

A read-only machine learning research tool that analyzes and predicts the direction (**UP**, **DOWN**, or **NO TRADE**) of 1-minute `EURUSD-OTC` candles on IQ Option using the [OTCharts](https://otcharts.com) API.

---

## 1. What This Project Does

* **Target Instrument:** 1-minute (`60s`) `EURUSD-OTC` candles from IQ Option (venue: `iq`).
* **Model Objective:** Predict whether the next completed candle will close **UP** or **DOWN**, or trigger **NO TRADE** if confidence, data quality, or calibration thresholds are not met.
* **Non-Trading / Read-Only:** It **never connects to a broker account and never places trades**.
* **Strict Anti-Illusion Safeguards:**
  * **Zero Lookahead Leakage:** Features for bar $t$ are computed strictly using completed bars $\le t$ to predict bar $t+1$.
  * **Forming Bar Isolation:** The in-progress bar is excluded from model inputs and drawn separately in grey.
  * **Strict Probability Calibration:** Employs out-of-fold Platt scaling. If Expected Calibration Error (ECE) exceeds the noise floor by $>0.02$, directional calls are blocked and gated to `NO TRADE`.
  * **Staleness & Gap Protection:** Automatically falls back to `NO TRADE` if feeds are stale ($>150\text{s}$) or if gaps occur within the 30-bar lookback window.
  * **DOJI Handling:** Candles where $|close - open| \le 10^{-5}$ are treated as ties, excluded from training, and never counted as wins.

---

## 2. Architecture & File Structure

| Directory / File | Description |
|---|---|
| `app.py` | Interactive Streamlit dashboard with live charts and real-time signals |
| `config.py` | Central configuration, thresholds, symbols, and database path settings |
| `api/` | REST adapters (OTCharts & keyless public Binance adapter) |
| `data/` | SQLite database layer (`predictor.db`), candle parser, and live collector |
| `features/` | Technical indicators (RSI, MACD, EMAs, Bollinger, ATR) and leakage-safe datasets |
| `models/` | Baseline classifiers (`majority`, `momentum`), ML models (`logreg`, `hgb`), and Platt calibrator |
| `evaluation/` | Walk-forward expanding-window backtesting, outcome matcher, and offline PnL simulator |
| `dashboard/` | Plotly candlestick charts, prediction cards, and metric panels |
| `scripts/` | Standalone CLI scripts: `collect_binance.py`, `train_model.py`, `run_backtest.py`, `seed_sample_data.py` |
| `tests/` | 55 unit tests covering leakage prevention, fixtures, math, and data pipelines |

---

## 3. Step-by-Step Setup & Run Guide

### Prerequisites
* **Python 3.10+** (Python 3.11 recommended)
* An API Key from [OTCharts](https://otcharts.com/account) *(optional for unit tests; required for live fetching)*

---

### Step 1: Create Virtual Environment & Install Dependencies

#### On macOS / Linux:
```bash
# Navigate to project folder
cd eurusd_otc_predictor

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### On Windows (PowerShell):
```powershell
cd eurusd_otc_predictor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2: Run Unit Tests (No API Key Required)
Run the test suite to verify the installation, synthetic fixtures, leakage checks, and models:

```bash
pytest -q
```
*(All 55 tests should pass).*

---

### Step 3: Configure OTCharts API Key
Create your `.env` file from the example template:

```bash
cp .env.example .env
```

Open `.env` in an editor and insert your key:
```env
OTCHARTS_API_KEY=otc_live_YOUR_ACTUAL_KEY_HERE
```

*Alternatively, set it in your current terminal session:*
* **macOS / Linux:** `export OTCHARTS_API_KEY="otc_live_YOUR_KEY"`
* **Windows (PowerShell):** `$env:OTCHARTS_API_KEY = "otc_live_YOUR_KEY"`

---

### Step 4: Verify API Connectivity & Rate Limits
Run the verification script to check your account quota and confirm instrument status:

```bash
python scripts/verify_api.py
```

---

### Step 5: Collect Historical Candle Data
Fetch candles from OTCharts and store them in the local SQLite database (`data_storage/predictor.db`):

```bash
python scripts/collect_data.py --bars 5000
```
> *Note: At least 2,000 completed candles are required for default model training.*

---

### Step 6: Train Models
Fit the baseline and machine learning models (`majority`, `momentum`, `logreg`, `hgb`) with probability calibration:

```bash
python scripts/train_model.py
```

---

### Step 7: Run Historical Walk-Forward Backtest
Run walk-forward cross-validation with statistical significance tests ($p$-values vs. 50% random walk):

```bash
python scripts/run_backtest.py
```

---

### Step 8: Launch the Interactive Dashboard
Start the Streamlit user interface:

```bash
streamlit run app.py
```

Open your browser to:
**http://localhost:8501**

---

## 4. Dashboard Features

* **Tab 1: Live**
  * Real-time Plotly candlestick chart with completed candles and the forming candle (grey).
  * Signal card showing direction (**UP**, **DOWN**, or **NO TRADE**), confidence probability, lag in seconds, and reason if gated.
  * Top feature contributions (for logistic regression).
* **Tab 2: Performance**
  * Forward-testing rolling accuracy tracking realized predictions against closing prices.
  * Summary cards of the latest walk-forward backtests.
* **Tab 3: Data & Export**
  * Overview of total stored candles and recent data-integrity / error events.
  * One-click CSV export of stored candles and predictions.
* **Tab 4: Train / Backtest**
  * Retrain all models directly from the UI.
  * Run on-demand walk-forward backtests without leaving the browser.

---

## 5. Important Disclaimers

1. **OTC Price Action:** Broker OTC assets are internally calculated and do not represent interbank spot forex. Indicator trends and standard market dynamics may not apply.
2. **Read-Only Research:** This software is intended strictly for data science and statistical modeling exploration. It does not provide trading advice or automated execution.
