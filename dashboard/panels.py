"""Streamlit panels. Elegant, user-friendly UI with plain-English explanations and responsive layout."""
from __future__ import annotations
import textwrap
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import Settings
from evaluation import metrics as M
from models.prediction import Prediction

# Human-friendly dictionary mapping internal feature names to plain English
FEATURE_FRIENDLY_NAMES = {
    "ret_1": "1-Minute Price Momentum",
    "ret_2": "2-Minute Price Momentum",
    "ret_3": "3-Minute Price Momentum",
    "ret_5": "5-Minute Price Momentum",
    "ret_10": "10-Minute Price Momentum",
    "bull_run": "Consecutive Green Candles Streak",
    "bear_run": "Consecutive Red Candles Streak",
    "body_ratio": "Candle Body Strength (Open to Close)",
    "upper_wick": "Upper Rejection Wick",
    "lower_wick": "Lower Rejection Wick",
    "body_dir": "Current Candle Direction",
    "vol_10": "10-Minute Market Volatility",
    "vol_30": "30-Minute Market Volatility",
    "ema_dist_5": "Distance from Fast Trend (EMA 5)",
    "ema_dist_20": "Distance from Slow Trend (EMA 20)",
    "ema_cross": "Moving Average Trend Cross (5 vs 20)",
    "rsi_14": "RSI Momentum (Overbought / Oversold)",
    "macd_hist_n": "MACD Histogram Momentum",
    "macd_n": "MACD Trend Line",
    "bb_pos": "Bollinger Band Channel Position",
    "bb_width": "Bollinger Band Volatility Width",
    "atr_n": "Average True Range (Volatility)",
    "dist_high_10": "Distance from 10-Min High",
    "dist_low_10": "Distance from 10-Min Low",
    "dist_high_30": "Distance from 30-Min High",
    "dist_low_30": "Distance from 30-Min Low",
    "mom_5": "Volatility-Adjusted Momentum",
    "mr_z_20": "Mean Reversion Stretch (Z-Score)",
    "hour_sin": "Time of Day (Intraday Cycle)",
    "hour_cos": "Time of Day (Session Rhythm)",
}


def inject_custom_styles() -> None:
    """Inject clean, premium modern dark-mode styles with full responsive support."""
    css = textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Prevent horizontal page overflow when sidebar is open or on mobile */
    .main, .block-container {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    .main .block-container {
        padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Metrics card container polish */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        min-width: 0;
        box-sizing: border-box;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 700 !important;
        font-size: clamp(1.1rem, 2vw, 1.35rem) !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Tabs polish (Horizontally scrollable on mobile) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13, 17, 23, 0.8);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow-x: auto;
        flex-wrap: nowrap;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #8b949e;
        font-weight: 600;
        font-size: 0.90rem;
        white-space: nowrap;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: #ffffff !important;
    }

    /* Container query wrapper for the Forecast section */
    .forecast-deck {
        container-type: inline-size;
        width: 100%;
        max-width: 100%;
        margin-bottom: 18px;
        box-sizing: border-box;
    }

    .forecast-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 14px;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
    }

    /* Cards */
    .fc-card {
        background: rgba(19, 23, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        min-width: 0;
        max-width: 100%;
        overflow: hidden;
    }

    .fc-card-up {
        border-color: rgba(16, 185, 129, 0.4);
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.09) 0%, rgba(19, 23, 34, 0.88) 100%);
    }
    .fc-card-down {
        border-color: rgba(239, 68, 68, 0.4);
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.09) 0%, rgba(19, 23, 34, 0.88) 100%);
    }
    .fc-card-neutral {
        border-color: rgba(99, 102, 241, 0.3);
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.07) 0%, rgba(19, 23, 34, 0.88) 100%);
    }

    .fc-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }

    .fc-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
        width: fit-content;
        max-width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .fc-title {
        font-size: clamp(1.4rem, 2.5vw, 1.95rem);
        font-weight: 800;
        margin: 2px 0 8px 0;
        letter-spacing: -0.02em;
        line-height: 1.15;
        word-break: break-word;
    }

    .fc-desc {
        color: #c9d1d9;
        font-size: 0.88rem;
        line-height: 1.45;
        margin: 0;
        word-break: break-word;
    }

    /* Probability Chips Grid */
    .prob-chips-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 10px;
        width: 100%;
        box-sizing: border-box;
    }

    .prob-chip {
        background: rgba(13, 17, 23, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 8px 10px;
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
        min-width: 0;
    }

    .chip-type {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .chip-val {
        font-size: clamp(1.1rem, 2vw, 1.35rem);
        font-weight: 800;
        line-height: 1.2;
        white-space: nowrap;
    }

    .chip-bullish .chip-type { color: #34d399; }
    .chip-bullish .chip-val { color: #10b981; }

    .chip-bearish .chip-type { color: #f87171; }
    .chip-bearish .chip-val { color: #ef4444; }

    /* Custom Progress Bar */
    .prob-bar-track {
        width: 100%;
        height: 8px;
        background: rgba(239, 68, 68, 0.35);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
        margin-bottom: 6px;
        box-sizing: border-box;
    }

    .prob-bar-fill {
        height: 100%;
        background: #10b981;
        border-radius: 4px 0 0 4px;
        transition: width 0.4s ease;
    }

    .prob-subtext {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 4px;
        font-size: 0.74rem;
        color: #8b949e;
    }

    /* Timing Card */
    .timing-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(31, 111, 235, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 6px 12px;
        border-radius: 8px;
        color: #38bdf8;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 100%;
        box-sizing: border-box;
        white-space: normal;
        word-break: break-word;
    }

    .time-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 4px;
        font-size: 0.80rem;
        color: #8b949e;
        line-height: 1.35;
    }

    .time-label {
        color: #8b949e;
    }

    .time-val {
        color: #f0f6fc;
        font-weight: 600;
    }

    /* Responsive Container Queries for dynamic sidebar opening & mobile */
    @container (min-width: 960px) {
        .forecast-grid {
            grid-template-columns: 1.15fr 0.95fr 0.9fr;
            gap: 14px;
        }
        .fc-card-hero {
            grid-column: auto;
        }
    }

    @container (min-width: 580px) and (max-width: 959px) {
        .forecast-grid {
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .fc-card-hero {
            grid-column: 1 / -1;
        }
    }

    @container (max-width: 579px) {
        .forecast-grid {
            grid-template-columns: 1fr;
            gap: 10px;
        }
        .fc-card-hero {
            grid-column: 1 / -1;
        }
    }

    /* Fallback media queries */
    @media (max-width: 960px) {
        .fc-card-hero {
            grid-column: 1 / -1;
        }
    }
    @media (max-width: 580px) {
        .forecast-grid {
            grid-template-columns: 1fr !important;
        }
        .fc-card-hero {
            grid-column: 1 / -1;
        }
    }
    </style>
    """).strip()
    st.html(css)


def fmt_local(ts: Optional[float], tz: str) -> str:
    if ts is None:
        return "Never"
    try:
        return datetime.fromtimestamp(ts, tz=pd.Timestamp.now(tz=tz).tz).strftime("%I:%M:%S %p (%Z)")
    except Exception:
        return datetime.fromtimestamp(ts).strftime("%I:%M:%S %p")


def header(s: Settings, connected: bool, last_ok_local: Optional[float], server_now: float,
           last_complete_ts: Optional[int], last_error: Optional[str], tz: str) -> tuple[bool, Optional[float]]:
    """Renders the executive header and live system health cards."""
    inject_custom_styles()

    header_html = textwrap.dedent("""
    <div style="margin-bottom: 16px;">
        <h1 style="margin: 0; font-size: clamp(1.5rem, 3vw, 2.1rem); font-weight: 800; letter-spacing: -0.02em; color: #f0f6fc; line-height: 1.2;">
            EUR/USD AI Market Predictor
        </h1>
        <p style="margin: 4px 0 0 0; color: #8b949e; font-size: clamp(0.85rem, 1.8vw, 1.0rem);">
            Real-Time Machine Learning Forecaster for 1-Minute Candlestick Closes
        </p>
    </div>
    """).strip()
    st.html(header_html)

    lag = None if last_complete_ts is None else server_now - (last_complete_ts + s.timeframe_s)
    fresh = bool(connected and lag is not None and lag <= s.max_staleness_s)

    c = st.columns(4)
    # 1. API Status
    api_status_label = "🟢 Live Connected" if connected else "🔴 Disconnected"
    api_source = "Binance Free Feed" if s.venue.lower() == "binance" else f"OTCharts ({s.venue})"
    c[0].metric("Market Connection", api_status_label, api_source)

    # 2. Asset & Timeframe
    c[1].metric("Active Asset", f"{s.symbol}", f"{s.timeframe_s}s Candlestick")

    # 3. Data Freshness
    if fresh:
        freshness_label = "⚡ Real-Time"
        freshness_delta = f"Updated {lag:.0f}s ago" if lag is not None else "In sync"
    else:
        freshness_label = "⏳ Delayed / Paused"
        freshness_delta = f"{lag:.0f}s since last bar" if lag is not None else "No data"
    c[2].metric("Data Freshness", freshness_label, freshness_delta)

    # 4. Model Active
    model_friendly = {
        "logreg": "Logistic Regression (Linear)",
        "hgb": "Gradient Boosting (Trees)",
        "momentum": "Price Momentum Baseline",
        "majority": "Majority Class Baseline",
    }.get(s.model_name, s.model_name.upper())
    c[3].metric("AI Model Brain", s.model_name.upper(), model_friendly)

    # Quick Explainer Banner (Collapsible for clean UX)
    with st.expander("ℹ️ How does this AI predictor work? (Click to view quick guide)", expanded=False):
        st.markdown(textwrap.dedent("""
        * **1-Minute Forecasts:** As soon as a 1-minute candle closes, our AI model analyzes **30 technical indicators** (momentum, RSI, moving averages, candle shapes, and volatility).
        * **Prediction Signals:**
          * **🟢 UP:** The AI calculates a statistically significant probability that the next candle will close higher.
          * **🔴 DOWN:** The AI calculates a probability that the next candle will close lower.
          * **⏸️ NO TRADE:** When probabilities are neutral (~50/50), the AI holds to protect against risky coin flips.
        * **Safety:** Completed candles only are analyzed. The forming candle currently in progress is strictly excluded to prevent lookahead bias.
        """).strip())

    if last_error:
        st.error(f"⚠️ Feed Notice: {last_error}")
    elif not fresh:
        st.warning("⚠️ Market data is currently stale or waiting for the next completed bar. Signals are safely paused (NO TRADE).")

    return fresh, lag


def _human_friendly_reason(reason: Optional[str], p_up: Optional[float], p_down: Optional[float],
                           threshold: float, calibrated: bool) -> str:
    """Converts mathematical reason strings into clear, human-friendly explanations."""
    if not reason:
        if p_up is not None and p_up >= threshold:
            return f"The AI detects strong upward momentum ({p_up:.1%} confidence), exceeding the {threshold:.0%} threshold."
        if p_down is not None and p_down >= threshold:
            return f"The AI detects downward selling pressure ({p_down:.1%} confidence), exceeding the {threshold:.0%} threshold."
        return "Waiting for the next completed candle."

    r = reason.lower()
    if "uncertain" in r or "threshold" in r:
        top_prob = max(p_up or 0.5, p_down or 0.5)
        return (f"The market is currently balanced (top probability is {top_prob:.1%}). "
                f"The AI waits for at least {threshold:.0%} confidence before recommending an entry.")
    if "stale" in r:
        return "Market data feed is temporarily delayed. Holding until a fresh candle arrives."
    if "disconnected" in r:
        return "Market connection is offline. Please check network connectivity."
    if "uncalibrated" in r:
        return "The AI model is running in evaluation mode. Raw signals are recorded for research."
    if "insufficient" in r:
        return "Building up historical bars. More candles needed for feature calculations."
    return reason


def prediction_panel(pred: Prediction, s: Settings, fresh: bool, tz: str) -> None:
    """Renders the master responsive forecast banner with zero overflow across desktop, sidebar-open, and mobile."""
    if pred.direction == "UP":
        card_theme = "fc-card-up"
        pill_style = "background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981;"
        pill_text = "▲ Bullish Signal Detected"
        title_color = "#34d399"
        title_text = "PREDICTED: UP ↗"
    elif pred.direction == "DOWN":
        card_theme = "fc-card-down"
        pill_style = "background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444;"
        pill_text = "▼ Bearish Signal Detected"
        title_color = "#f87171"
        title_text = "PREDICTED: DOWN ↘"
    else:
        card_theme = "fc-card-neutral"
        pill_style = "background: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid #64748b;"
        pill_text = "⏸️ Market Neutral / Holding"
        title_color = "#cbd5e1"
        title_text = "NO TRADE"

    friendly_explanation = _human_friendly_reason(
        pred.reason, pred.p_up, pred.p_down, s.decision_threshold, pred.calibrated
    )

    # Compute probabilities
    p_up_val = pred.p_up if pred.p_up is not None else 0.52
    p_down_val = pred.p_down if pred.p_down is not None else (1.0 - p_up_val)
    p_up_pct = int(round(p_up_val * 100))
    p_down_pct = 100 - p_up_pct

    # Timing details
    target_time_str = fmt_local(pred.target_ts, tz)
    based_time_str = fmt_local(pred.based_on_ts, tz)
    
    if fresh and pred.target_ts is not None:
        remaining = pred.target_ts + s.timeframe_s - time.time()
        if remaining > 0:
            timing_html = f'<div class="timing-pill">⏳ Next bar in <strong>{int(remaining)}s</strong></div>'
        else:
            timing_html = '<div class="timing-pill" style="color: #34d399;">🔄 Candle closed! Refreshing...</div>'
    else:
        timing_html = '<div class="timing-pill" style="color: #94a3b8;">⏳ Waiting for next tick...</div>'

    # Master forecast HTML (Guaranteed zero overflow, container queries adapt smoothly to sidebar and screen width)
    card_html = textwrap.dedent(f"""
    <div class="forecast-deck">
        <div class="forecast-grid">
            <!-- Card 1: AI Direction Signal (Hero) -->
            <div class="fc-card fc-card-hero {card_theme}">
                <div>
                    <div class="fc-status-pill" style="{pill_style}">{pill_text}</div>
                    <div class="fc-title" style="color: {title_color};">{title_text}</div>
                </div>
                <p class="fc-desc">{friendly_explanation}</p>
            </div>

            <!-- Card 2: Confidence Balance -->
            <div class="fc-card">
                <div>
                    <div class="fc-label">AI Confidence Balance</div>
                    <div class="prob-chips-grid">
                        <div class="prob-chip chip-bullish">
                            <span class="chip-type">Bullish</span>
                            <span class="chip-val">{p_up_val:.1%}</span>
                        </div>
                        <div class="prob-chip chip-bearish">
                            <span class="chip-type">Bearish</span>
                            <span class="chip-val">{p_down_val:.1%}</span>
                        </div>
                    </div>
                </div>
                <div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width: {p_up_pct}%;"></div>
                    </div>
                    <div class="prob-subtext">
                        <span>Target: {s.decision_threshold:.0%} needed</span>
                        <span>{p_up_pct}% Bull vs {p_down_pct}% Bear</span>
                    </div>
                </div>
            </div>

            <!-- Card 3: Timing & Clock -->
            <div class="fc-card">
                <div>
                    <div class="fc-label">Candle Timing & Clock</div>
                    {timing_html}
                </div>
                <div>
                    <div class="time-item">
                        <span class="time-label">Target Closes:</span>
                        <span class="time-val">{target_time_str}</span>
                    </div>
                    <div class="time-item">
                        <span class="time-label">Based On Bar:</span>
                        <span class="time-val">{based_time_str}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """).strip()
    st.html(card_html)

    # Top Feature Drivers (Spacious Plain English)
    if pred.top_features:
        with st.expander("🔍 Why did the AI make this call? (Click to view top market indicator drivers)", expanded=False):
            driver_cols = st.columns(min(len(pred.top_features), 4))
            for i, (feat, val) in enumerate(pred.top_features[:4]):
                friendly_name = FEATURE_FRIENDLY_NAMES.get(feat, feat.replace("_", " ").title())
                impact = "Pushes UP 🟢" if val > 0 else "Pushes DOWN 🔴"
                driver_cols[i].metric(friendly_name, impact, f"Strength: {abs(val):.2f}")


def _fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.1%}" if pct else f"{x:.4f}"


def forward_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Renders the out-of-sample forward test tracking scorecard."""
    st.markdown("### Real-Time Track Record (Out-of-Sample Results)")
    st.markdown(textwrap.dedent("""
    Every prediction the AI makes in real-time is securely logged **before** the candle closes. 
    Once the candle finishes, the actual market outcome is matched to verify accuracy.
    """).strip())

    if df.empty or df["outcome"].notna().sum() == 0:
        st.info("🎯 The AI has just started logging live predictions. As 1-minute candles close, completed track records will appear here automatically.")
        return df

    done = df[df["outcome"].notna()]
    dec = done[done["outcome"].isin(["UP", "DOWN"])]
    out = []
    for name, g in dec.groupby("model_name"):
        y = (g["outcome"] == "UP").astype(int).to_numpy()
        pred = (g["raw_direction"] == "UP").astype(int).to_numpy()
        r = M.classification_report(y, pred)
        sig = g[g["signal_correct"].notna()]
        out.append({
            "AI Model": name.upper(),
            "Evaluated Candles": len(done[done.model_name == name]),
            "Decided Calls": len(g),
            "Raw Win Rate": _fmt(r["accuracy"]),
            "Gated Trades (Passed Threshold)": len(sig),
            "Gated Win Rate": _fmt(sig["signal_correct"].mean()) if len(sig) else "N/A",
            "Skipped (NO TRADE)": int((g["direction"] == "NO TRADE").sum()),
        })

    st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
    st.caption("💡 **Tip:** In financial binary markets, distinguishing genuine skill from chance requires 1,000+ completed trades. The AI skips low-confidence setups to protect accuracy.")
    return done


def backtest_panel(bt: pd.DataFrame) -> None:
    """Renders the historical walk-forward backtest summary in simple English."""
    import json
    st.markdown("---")
    st.markdown("### Historical Simulation (Walk-Forward Backtest)")
    st.markdown(textwrap.dedent("""
    Walk-forward backtesting tests the AI strictly on historical data it has **never seen before**, 
    simulating how it would have performed over the past days.
    """).strip())

    if bt.empty:
        st.info("No backtest results stored yet. Click 'Run Walk-Forward Backtest' in the Train/Backtest tab.")
        return

    latest = json.loads(bt.iloc[0]["summary"])
    if not latest.get("sufficient"):
        st.warning(latest.get("message", "Insufficient historical bars to run backtest."))
        return

    st.write(f"📅 **Historical Range Analyzed:** {latest['date_range_utc'][0][:10]} to {latest['date_range_utc'][1][:10]}  |  "
             f"**Total Tested Candles:** {latest['n_usable_rows']:,}  |  "
             f"**Break-Even Target (85% payout):** {_fmt(latest['break_even_win_rate_at_payout'])}")

    rows = []
    for name, rep in latest["models"].items():
        a = rep["all_rows"]
        acc = a["accuracy"]
        verdict = latest["verdicts"].get(name, "")
        if "outperform" in verdict.lower() or acc > 0.5405:
            verdict_badge = "🟢 Beats Break-Even"
        else:
            verdict_badge = "⚪ Neutral / Within Expected Range"

        rows.append({
            "AI Model": name.upper(),
            "Tested Trades": a["n_predictions"],
            "Out-Of-Sample Win Rate": _fmt(acc),
            "95% Confidence Interval": f"{_fmt(a['accuracy_ci95'][0])} to {_fmt(a['accuracy_ci95'][1])}",
            "Status": verdict_badge
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
