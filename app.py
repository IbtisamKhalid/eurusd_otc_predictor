"""EUR/USD AI Market Predictor -- Streamlit Dashboard."""
from __future__ import annotations
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from api.client import OTChartsClient
from api.schemas import ConfigError
from config import DB_PATH, Settings
from data.collector import Collector
from data.database import Database
from dashboard import charts, panels
from evaluation.backtest import run_backtest, summary_for_storage
from evaluation.forward import match_outcomes, record_prediction
from features.engineering import build_dataset, trainable
from models.prediction import make_prediction
from models.training import MODEL_NAMES, load_model, save_model, train_model

MODEL_DIR = config.BASE_DIR / "data_storage" / "models"
st.set_page_config(
    page_title="EUR/USD AI Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_db() -> Database:
    return Database(DB_PATH)


def sidebar() -> tuple[Settings, str, str]:
    st.sidebar.markdown("### ⚙️ System Controls")
    
    # Provider selection
    provider_choice = st.sidebar.radio(
        "Market Feed Source",
        ["Binance (Real Live Market - 24/7)", "IQ Option / OTCharts (OTC)"],
        index=0 if config.VENUE == "binance" else 1,
        help="Binance provides free live 24/7 market data without requiring any API keys. IQ Option uses OTC feeds."
    )
    is_binance = "Binance" in provider_choice
    venue_val = "binance" if is_binance else "iq"
    default_sym = "EURUSDT" if is_binance else "EURUSD-OTC"

    symbol_val = st.sidebar.text_input(
        "Trading Pair Symbol",
        default_sym,
        help="e.g. EURUSDT or BTCUSDT for Binance; EURUSD-OTC for IQ Option."
    )

    timeframe_val = int(st.sidebar.selectbox(
        "Candle Timeframe",
        [60, 300],
        index=0,
        format_func=lambda x: "1 Minute (60s)" if x == 60 else "5 Minutes (300s)"
    ))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Brain & Thresholds")

    model_display_names = {
        "logreg": "Logistic Regression (Linear)",
        "hgb": "Gradient Boosting (Recommended)",
        "momentum": "Momentum Baseline",
        "majority": "Majority Class Baseline"
    }
    model_choice = st.sidebar.selectbox(
        "Active AI Model",
        MODEL_NAMES,
        index=3 if "hgb" in MODEL_NAMES else 2,
        format_func=lambda x: model_display_names.get(x, x.upper())
    )

    threshold_val = st.sidebar.slider(
        "Minimum Decision Confidence",
        0.50, 0.75, 0.55, 0.01,
        format="%.0f%%",
        help="The AI only outputs an UP or DOWN signal if confidence reaches this level. Below this, it prudently chooses NO TRADE."
    )

    candles_shown = st.sidebar.slider("Chart Candles Displayed", 30, 300, 100)
    chart_engine = st.sidebar.radio(
        "Interactive Chart Engine",
        ["⚡ TradingView (Ultra-Fast 60fps)", "📊 Plotly (Classic Heavy)"],
        index=0,
        help="TradingView loads in <0.5s over public links with genuine 60fps TradingView canvas rendering."
    )
    tz = st.sidebar.text_input("Timezone Display", "Asia/Kuala_Lumpur")

    s = Settings(
        venue=venue_val,
        symbol=symbol_val,
        timeframe_s=timeframe_val,
        poll_interval_s=60,
        max_staleness_s=150,
        model_name=model_choice,
        decision_threshold=threshold_val,
        min_train_samples=2000,
        retention_days=90,
        display_candles=candles_shown,
    )
    return s, tz, chart_engine


def get_collector(s: Settings, db: Database):
    key = (s.venue, s.symbol, s.timeframe_s)
    if st.session_state.get("collector_key") != key:
        try:
            if s.venue.lower() == "binance":
                from api.binance_client import BinanceClient
                st.session_state["collector"] = Collector(BinanceClient(), db, s)
            else:
                st.session_state["collector"] = Collector(OTChartsClient(), db, s)
            st.session_state["collector_err"] = None
        except ConfigError as exc:
            st.session_state["collector"], st.session_state["collector_err"] = None, str(exc)
        st.session_state["collector_key"] = key
    col = st.session_state["collector"]
    if col is not None:
        col.s = s
    return col


def main() -> None:
    s, tz, chart_engine = sidebar()
    db = get_db()
    col = get_collector(s, db)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Auto-Update")
    auto = st.sidebar.checkbox("Auto-Refresh Every 60s", value=True)
    if st.sidebar.button("🔄 Check & Fetch Now"):
        st.session_state["last_fetch"] = 0.0
        st.rerun()

    if st.session_state.get("collector_err"):
        st.error(st.session_state["collector_err"])

    last_fetch = st.session_state.get("last_fetch", 0.0)
    if col is not None and (time.time() - last_fetch >= s.poll_interval_s or "last_fetch" not in st.session_state):
        res = col.fetch_and_store(limit=300)
        st.session_state["last_fetch"] = time.time()
        st.session_state["last_result"] = res
        db.purge_older_than(s.retention_days, s.venue, s.symbol, s.timeframe_s)

    candles = db.load_candles(s.venue, s.symbol, s.timeframe_s)
    state = col.state if col else None
    server_now = state.server_now() if state else time.time()
    last_complete = int(candles["ts"].iloc[-1]) if len(candles) else None

    # Top Header & Status Cards
    fresh, lag = panels.header(
        s,
        bool(state and state.connected),
        state.last_success_local if state else None,
        server_now,
        last_complete,
        state.last_error if state else None,
        tz
    )

    # Resolve past predictions whose target candles have closed
    match_outcomes(db, s)

    # Compute prediction for the selected model
    shown = None
    for name in MODEL_NAMES:
        tm = load_model(MODEL_DIR, name)
        if tm is None or len(candles) == 0:
            continue
        p = make_prediction(candles, tm, s, server_now, bool(state and state.connected))
        if fresh:
            record_prediction(db, p, s)
        if name == s.model_name:
            shown = p

    if shown is None:
        shown = make_prediction(candles, None, s, server_now, bool(state and state.connected))

    # --- TABS ---
    tabs = st.tabs([
        "📈 Live Market & Forecast",
        "🏆 Track Record & Accuracy",
        "🧪 AI Training & Backtest",
        "💾 Data History & Export"
    ])

    # TAB 1: Live Market & Forecast
    with tabs[0]:
        # 1. TOP SECTION: AI Forecast & Explanation (Full Width)
        panels.prediction_panel(shown, s, fresh, tz)

        st.html("<div style='height: 14px;'></div>")

        # 2. BOTTOM SECTION: Full-Width Professional Trading Chart
        c_head, c_nav = st.columns([1, 1])
        with c_head:
            st.markdown("### 📊 Live Interactive Market Chart")
        with c_nav:
            range_sel = st.radio(
                "Quick Time Window",
                options=["30m", "1h", "2h", "4h", "Full History"],
                index=1,
                horizontal=True,
                label_visibility="collapsed"
            )

        range_map = {
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "Full History": min(len(candles), 1000)
        }
        n_disp = range_map.get(range_sel, 60)

        if len(candles):
            preds = db.predictions_with_outcomes()
            disp = candles.tail(n_disp)
            
            if chart_engine.startswith("⚡"):
                import streamlit.components.v1 as components
                tv_html = charts.tradingview_candlestick(
                    disp,
                    state.forming if state else None,
                    preds if len(preds) else None,
                    shown.target_ts,
                    tz,
                    height=520
                )
                components.html(tv_html, height=530)
                st.caption(
                    "⚡ **TradingView Lightweight Engine:** Drag chart left/right to pan history • "
                    "Scroll wheel / pinch to zoom • Live OHLC & EMA crosshair legend • Instant load"
                )
            else:
                fig = charts.candlestick(
                    disp,
                    state.forming if state else None,
                    preds if len(preds) else None,
                    shown.target_ts,
                    tz
                )
                chart_config = {
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToAdd": [
                        "drawline",
                        "eraseshape",
                        "zoomIn2d",
                        "zoomOut2d",
                        "autoScale2d",
                        "resetScale2d"
                    ],
                    "displaylogo": False,
                }
                st.plotly_chart(fig, use_container_width=True, config=chart_config)
                st.caption(
                    "🖐️ **Navigation Controls:** **Click & drag on the chart** to move backward & forward in time. "
                    "**Scroll wheel / pinch trackpad** to zoom in & out smoothly. Price scale is on the right."
                )
        else:
            st.info("No candles found in database for this pair yet. Click 'Check & Fetch Now' in the sidebar.")

    # TAB 2: Track Record & Accuracy
    with tabs[1]:
        df = db.predictions_with_outcomes()
        done = panels.forward_panel(df)
        if len(done):
            sel = done[done["model_name"] == s.model_name]
            if len(sel):
                if chart_engine.startswith("⚡"):
                    d_acc = sel[sel["raw_correct"].notna()].copy()
                    if len(d_acc):
                        d_acc["Win Rate (%)"] = d_acc["raw_correct"].astype(float).rolling(50, min_periods=10).mean() * 100
                        d_acc["Break-Even Target (54.1%)"] = 54.05
                        d_acc["Coin Toss Baseline (50%)"] = 50.0
                        d_acc["Time"] = pd.to_datetime(d_acc["target_ts"], unit="s", utc=True).dt.tz_convert(tz).dt.strftime("%I:%M %p")
                        chart_data = d_acc.set_index("Time")[["Win Rate (%)", "Break-Even Target (54.1%)", "Coin Toss Baseline (50%)"]]
                        st.line_chart(chart_data, height=320)
                else:
                    st.plotly_chart(charts.rolling_accuracy(sel, "raw_correct"), use_container_width=True)
        panels.backtest_panel(db.latest_backtests())

    # TAB 3: AI Training & Backtest
    with tabs[2]:
        st.markdown("### Model Retraining & Historical Simulation")
        st.markdown(
            "You can retrain all machine learning models on stored market candles, or run a walk-forward "
            "historical backtest to see how the strategy would have performed over the past days."
        )
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Retrain All Models on Current Data", use_container_width=True):
                with st.spinner("Retraining AI models..."):
                    ds = trainable(build_dataset(candles, s.timeframe_s, s.doji_tolerance)) if len(candles) else pd.DataFrame()
                    try:
                        for name in MODEL_NAMES:
                            tm = train_model(ds, name, s.min_train_samples)
                            save_model(tm, MODEL_DIR)
                            db.save_model_version(tm.version, tm.name, tm.n_train, tm.calibration.status, tm.info)
                        st.success(f"✅ Successfully retrained all 4 models on {len(ds)} usable rows!")
                        st.rerun()
                    except ValueError as exc:
                        st.error(f"Cannot train: {exc}")

        with col_btn2:
            if st.button("📊 Run Walk-Forward Historical Simulation", use_container_width=True):
                with st.spinner("Simulating past performance..."):
                    res = run_backtest(candles, s)
                    summ = summary_for_storage(res)
                    db.save_backtest("all", 0, 0, res.get("n_usable_rows", 0), summ)
                if res.get("sufficient"):
                    st.success("✅ Backtest complete! See the updated results in the 'Track Record' tab.")
                    st.rerun()
                else:
                    st.warning(res.get("message"))

    # TAB 4: Data History & Export
    with tabs[3]:
        st.markdown("### Database & Data Integrity")
        st.write(f"📁 **Total Completed Candles in Database:** **{len(candles):,}**")
        
        ev = db.recent_events(20)
        st.markdown("#### Recent Data Quality & System Events")
        if len(ev):
            st.dataframe(ev, hide_index=True, use_container_width=True)
        else:
            st.info("No data issues or errors logged. Clean data feed.")

        out = config.BASE_DIR / "data_storage"
        if st.button("📥 Export Candles & Predictions to CSV"):
            n1 = db.export_csv("candles", out / "candles_export.csv")
            n2 = db.export_csv("predictions", out / "predictions_export.csv")
            st.success(f"Wrote {n1} candles and {n2} predictions to {out}")

    if auto:
        time.sleep(min(s.poll_interval_s, 60))
        st.rerun()


main()
