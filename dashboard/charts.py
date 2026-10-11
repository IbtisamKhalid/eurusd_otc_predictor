"""Plotly figures. Professional TradingView-style financial chart styling."""
from __future__ import annotations
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.schemas import Candle

TV_BG = "#131722"           # TradingView signature dark background
GRID = "rgba(255, 255, 255, 0.05)"
UP_COLOR = "#089981"         # TradingView emerald green
DOWN_COLOR = "#f23645"       # TradingView ruby red
FORMING_COLOR = "#787b86"    # Neutral grey
EMA5_COLOR = "#2962ff"       # Fast trend (electric blue)
EMA20_COLOR = "#ff6d00"      # Slow trend (warm amber)


def candlestick(candles: pd.DataFrame, forming: Optional[list[Candle]], preds: Optional[pd.DataFrame],
                target_ts: Optional[int], tz: str, warn_gaps: bool = True) -> go.Figure:
    """Create a 2-row professional TradingView-style chart with pan navigation:

    Row 1 (82%): Candlesticks + EMA 5 + EMA 20 + AI Signal markers
    Row 2 (18%): Volume bars colored by candle direction
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.82, 0.18]
    )

    d = candles.copy()
    d["t"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(tz)

    # Calculate moving averages for trend overlays
    d["ema5"] = d["close"].ewm(span=5, adjust=False).mean()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()

    # 1. Standard Financial Candlesticks (Row 1)
    # Using TradingView solid green / solid red style with crisp 1.2px wicks
    fig.add_trace(
        go.Candlestick(
            x=d["t"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="EUR/USDT 1m",
            increasing=dict(
                line=dict(color=UP_COLOR, width=1.2),
                fillcolor=UP_COLOR
            ),
            decreasing=dict(
                line=dict(color=DOWN_COLOR, width=1.2),
                fillcolor=DOWN_COLOR
            ),
            whiskerwidth=0,          # Removes boxplot horizontal T-caps for true financial wicks
            showlegend=True,
        ),
        row=1, col=1
    )

    # 2. Moving Average Overlays (Row 1)
    fig.add_trace(
        go.Scatter(
            x=d["t"],
            y=d["ema5"],
            mode="lines",
            name="Fast Trend (5 EMA)",
            line=dict(color=EMA5_COLOR, width=1.5),
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=d["t"],
            y=d["ema20"],
            mode="lines",
            name="Slow Trend (20 EMA)",
            line=dict(color=EMA20_COLOR, width=1.5),
            hoverinfo="skip"
        ),
        row=1, col=1
    )

    # 3. Forming In-Progress Candle (Row 1)
    if forming:
        completed_ts_set = set(d["ts"])
        forming_clean = [c for c in forming if c.ts not in completed_ts_set]
        if forming_clean:
            f = pd.DataFrame([c.__dict__ for c in forming_clean])
            f["t"] = pd.to_datetime(f["ts"], unit="s", utc=True).dt.tz_convert(tz)
            fig.add_trace(
                go.Candlestick(
                    x=f["t"],
                    open=f["open"],
                    high=f["high"],
                    low=f["low"],
                    close=f["close"],
                    name="Forming Bar (Closing Soon)",
                    increasing=dict(
                        line=dict(color=FORMING_COLOR, width=1.2),
                        fillcolor=FORMING_COLOR
                    ),
                    decreasing=dict(
                        line=dict(color=FORMING_COLOR, width=1.2),
                        fillcolor=FORMING_COLOR
                    ),
                    whiskerwidth=0,
                    opacity=0.6,
                    showlegend=True
                ),
                row=1, col=1
            )

    # 4. Past AI Prediction Markers (Row 1)
    if preds is not None and len(preds):
        p = preds[preds["target_ts"].isin(candles["ts"])].copy()
        p = p[p["direction"].isin(["UP", "DOWN"])]
        if len(p):
            hi = candles.set_index("ts")["high"]
            lo = candles.set_index("ts")["low"]
            p["t"] = pd.to_datetime(p["target_ts"], unit="s", utc=True).dt.tz_convert(tz)
            for direction, sym, color, ref in (("UP", "triangle-up", "#38bdf8", lo),
                                               ("DOWN", "triangle-down", "#fbbf24", hi)):
                q = p[p["direction"] == direction]
                if len(q):
                    y = q["target_ts"].map(ref) * (0.99995 if direction == "UP" else 1.00005)
                    ok = q["signal_correct"].map({1: "✅ Won", 0: "❌ Lost"}).fillna("⏳ Pending")
                    fig.add_trace(
                        go.Scatter(
                            x=q["t"],
                            y=y,
                            mode="markers",
                            name=f"AI {direction}",
                            marker=dict(symbol=sym, size=10, color=color, line=dict(width=1, color="#ffffff")),
                            text=ok,
                            hovertemplate="AI Call: " + direction + "<br>Outcome: %{text}<extra></extra>"
                        ),
                        row=1, col=1
                    )

    # 5. Target Candle Line (Current Forecast Horizon)
    if target_ts is not None:
        x = pd.to_datetime(target_ts, unit="s", utc=True).tz_convert(tz)
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color="#f59e0b",
            line_width=2,
            annotation_text="🎯 Target Candle",
            annotation_font=dict(color="#f59e0b", size=11),
            annotation_position="top left",
            row=1, col=1
        )

    # 6. Volume Subplot (Row 2)
    if "volume" in d.columns and d["volume"].notna().any():
        vol_colors = [UP_COLOR if c >= o else DOWN_COLOR for o, c in zip(d["open"], d["close"])]
        fig.add_trace(
            go.Bar(
                x=d["t"],
                y=d["volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.55,
                showlegend=False,
                hovertemplate="Volume: %{y:,.0f}<extra></extra>"
            ),
            row=2, col=1
        )

    # 7. Navigation, Layout & TradingView Crosshairs
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_BG,
        height=640,
        margin=dict(l=10, r=65, t=35, b=10),
        xaxis_rangeslider_visible=False,
        dragmode="pan",                       # Pan (drag to scroll left/right back and front in time)
        uirevision="tradingview_chart_state", # Preserves user pan & zoom across page refreshes
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.04,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color="#c9d1d9")
        ),
    )

    # Price Axis (Row 1): Right-aligned price scale with crosshair snapping
    fig.update_xaxes(
        gridcolor=GRID,
        showgrid=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="dot",
        spikecolor="#787b86",
        row=1, col=1
    )
    fig.update_yaxes(
        gridcolor=GRID,
        showgrid=True,
        tickformat=".5f",
        side="right",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="dot",
        spikecolor="#787b86",
        row=1, col=1
    )

    # Volume Axis (Row 2): Right-aligned, subtle
    fig.update_xaxes(
        gridcolor=GRID,
        showgrid=True,
        row=2, col=1
    )
    fig.update_yaxes(
        gridcolor=GRID,
        showgrid=False,
        side="right",
        showticklabels=False,
        row=2, col=1
    )

    return fig


def rolling_accuracy(df: pd.DataFrame, col: str, window: int = 50) -> go.Figure:
    d = df[df[col].notna()].copy()
    fig = go.Figure()
    if len(d):
        d["roll"] = d[col].astype(float).rolling(window, min_periods=min(10, window)).mean()
        d["t"] = pd.to_datetime(d["target_ts"], unit="s", utc=True)
        fig.add_trace(go.Scatter(
            x=d["t"],
            y=d["roll"],
            mode="lines",
            name=f"Rolling {window}-Trade Win Rate",
            line=dict(color="#38bdf8", width=2.5)
        ))
        fig.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="50% Coin Toss Baseline",
            annotation_font=dict(color="#94a3b8", size=11)
        )
        fig.add_hline(
            y=0.5405,
            line_dash="dash",
            line_color="#10b981",
            annotation_text="54.1% Break-Even Target (85% Payout)",
            annotation_font=dict(color="#10b981", size=11)
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_BG,
        height=320,
        dragmode="pan",
        uirevision="accuracy_chart_state",
        margin=dict(l=15, r=60, t=25, b=15),
        yaxis=dict(range=[0.35, 0.70], tickformat=".1%", gridcolor=GRID, side="right"),
        xaxis=dict(gridcolor=GRID),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def tradingview_candlestick(candles: pd.DataFrame, forming: Optional[list[Candle]] = None,
                            preds: Optional[pd.DataFrame] = None, target_ts: Optional[int] = None,
                            tz: str = "Asia/Kuala_Lumpur", height: int = 500) -> str:
    """Generates ultra-lightweight (20KB), high-performance TradingView chart HTML.
    
    Loads official TradingView Lightweight Charts directly from global edge CDN.
    Reduces data transfer across tunnels by 99.5% compared to heavy Plotly bundles.
    """
    import json
    d = candles.copy()
    d["ema5"] = d["close"].ewm(span=5, adjust=False).mean()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()

    candle_list = []
    volume_list = []
    ema5_list = []
    ema20_list = []

    for _, row in d.iterrows():
        t = int(row["ts"])
        o = round(float(row["open"]), 5)
        h = round(float(row["high"]), 5)
        l = round(float(row["low"]), 5)
        c = round(float(row["close"]), 5)
        v = round(float(row.get("volume", 0)), 2)
        candle_list.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        volume_list.append({
            "time": t,
            "value": v,
            "color": "rgba(8, 153, 129, 0.45)" if c >= o else "rgba(242, 54, 69, 0.45)"
        })
        if not pd.isna(row["ema5"]):
            ema5_list.append({"time": t, "value": round(float(row["ema5"]), 5)})
        if not pd.isna(row["ema20"]):
            ema20_list.append({"time": t, "value": round(float(row["ema20"]), 5)})

    if forming:
        completed_ts_set = set(d["ts"])
        for f in forming:
            if f.ts not in completed_ts_set:
                candle_list.append({
                    "time": int(f.ts),
                    "open": round(float(f.open), 5),
                    "high": round(float(f.high), 5),
                    "low": round(float(f.low), 5),
                    "close": round(float(f.close), 5),
                })

    markers = []
    if preds is not None and len(preds):
        completed_ts = set(d["ts"])
        p = preds[preds["target_ts"].isin(completed_ts)].copy()
        for _, row in p.iterrows():
            dir_val = row["direction"]
            if dir_val in ("UP", "DOWN"):
                is_up = dir_val == "UP"
                sig_cor = row.get("signal_correct")
                status = "Won" if sig_cor == 1 else "Lost" if sig_cor == 0 else "Pending"
                markers.append({
                    "time": int(row["target_ts"]),
                    "position": "belowBar" if is_up else "aboveBar",
                    "color": "#38bdf8" if is_up else "#fbbf24",
                    "shape": "arrowUp" if is_up else "arrowDown",
                    "text": f"AI {dir_val} ({status})"
                })
        markers.sort(key=lambda x: x["time"])

    candles_json = json.dumps(candle_list)
    volume_json = json.dumps(volume_list)
    ema5_json = json.dumps(ema5_list)
    ema20_json = json.dumps(ema20_list)
    markers_json = json.dumps(markers)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{
            margin: 0; padding: 0; width: 100%; height: 100%;
            background-color: #131722; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
        }}
        #chart-container {{
            width: 100%; height: {height}px; position: relative;
        }}
        .chart-header {{
            position: absolute; top: 10px; left: 14px; z-index: 10;
            display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
            background: rgba(19, 23, 34, 0.85); backdrop-filter: blur(6px);
            padding: 5px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);
            font-size: 11.5px; color: #8b949e; pointer-events: none;
        }}
        .symbol-badge {{
            color: #f0f6fc; font-weight: 700; font-size: 12.5px;
        }}
        .ohlc-group span {{ font-weight: 600; margin-left: 2px; }}
        .ohlc-up {{ color: #089981 !important; }}
        .ohlc-down {{ color: #f23645 !important; }}
        .ema5-badge {{ color: #2962ff; font-weight: 600; }}
        .ema20-badge {{ color: #ff6d00; font-weight: 600; }}
        .nav-help {{
            position: absolute; bottom: 8px; left: 14px; z-index: 10;
            font-size: 10px; color: #6e7681; background: rgba(19, 23, 34, 0.6);
            padding: 2px 6px; border-radius: 4px; pointer-events: none;
        }}
    </style>
</head>
<body>
    <div id="chart-container">
        <div class="chart-header">
            <span class="symbol-badge">EUR/USDT 1m</span>
            <span class="ohlc-group" id="ohlc-vals"></span>
            <span class="ema5-badge">EMA 5: <span id="val-ema5">-</span></span>
            <span class="ema20-badge">EMA 20: <span id="val-ema20">-</span></span>
        </div>
        <div class="nav-help">💡 Drag left/right to pan history • Scroll or pinch to zoom</div>
    </div>
    <script>
        const container = document.getElementById('chart-container');
        const chart = LightweightCharts.createChart(container, {{
            width: container.clientWidth,
            height: {height},
            layout: {{
                background: {{ color: '#131722' }},
                textColor: '#8b949e',
                fontSize: 11,
            }},
            grid: {{
                vertLines: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                horzLines: {{ color: 'rgba(255, 255, 255, 0.04)' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {{ color: '#787b86', width: 1, style: 1 }},
                horzLine: {{ color: '#787b86', width: 1, style: 1 }},
            }},
            rightPriceScale: {{
                borderColor: 'rgba(255, 255, 255, 0.08)',
                scaleMargins: {{ top: 0.08, bottom: 0.22 }},
                autoScale: true,
            }},
            timeScale: {{
                borderColor: 'rgba(255, 255, 255, 0.08)',
                timeVisible: true,
                secondsVisible: false,
                barSpacing: 9,
                minBarSpacing: 3,
            }},
            handleScroll: {{
                mouseWheel: true,
                pressedMouseMove: true,
                horzTouchDrag: true,
                vertTouchDrag: false
            }},
            handleScale: {{
                axisPressedMouseMove: true,
                mouseWheel: true,
                pinch: true
            }},
        }});

        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#089981',
            downColor: '#f23645',
            borderVisible: false,
            wickUpColor: '#089981',
            wickDownColor: '#f23645',
            priceFormat: {{ type: 'price', precision: 5, minMove: 0.00001 }},
        }});

        const volumeSeries = chart.addHistogramSeries({{
            color: '#26a69a',
            priceFormat: {{ type: 'volume' }},
            priceScaleId: '',
            scaleMargins: {{ top: 0.82, bottom: 0 }},
        }});

        const ema5Series = chart.addLineSeries({{
            color: '#2962ff',
            lineWidth: 2,
            priceScaleId: 'right',
            crosshairMarkerVisible: false,
        }});

        const ema20Series = chart.addLineSeries({{
            color: '#ff6d00',
            lineWidth: 2,
            priceScaleId: 'right',
            crosshairMarkerVisible: false,
        }});

        const candleData = {candles_json};
        const volumeData = {volume_json};
        const ema5Data = {ema5_json};
        const ema20Data = {ema20_json};
        const markersData = {markers_json};

        candleSeries.setData(candleData);
        volumeSeries.setData(volumeData);
        ema5Series.setData(ema5Data);
        ema20Series.setData(ema20Data);
        if (markersData && markersData.length) {{
            candleSeries.setMarkers(markersData);
        }}

        const ohlcEl = document.getElementById('ohlc-vals');
        const ema5El = document.getElementById('val-ema5');
        const ema20El = document.getElementById('val-ema20');

        function setLastBarLegend() {{
            if (candleData.length) {{
                const last = candleData[candleData.length - 1];
                updateLegend(last, ema5Data[ema5Data.length - 1]?.value, ema20Data[ema20Data.length - 1]?.value);
            }}
        }}

        function updateLegend(bar, e5, e20) {{
            if (!bar) return;
            const isUp = bar.close >= bar.open;
            const cls = isUp ? 'ohlc-up' : 'ohlc-down';
            ohlcEl.innerHTML = `O:<span class="${{cls}}">${{bar.open.toFixed(5)}}</span> H:<span class="${{cls}}">${{bar.high.toFixed(5)}}</span> L:<span class="${{cls}}">${{bar.low.toFixed(5)}}</span> C:<span class="${{cls}}">${{bar.close.toFixed(5)}}</span>`;
            if (e5 !== undefined) ema5El.innerText = e5.toFixed(5);
            if (e20 !== undefined) ema20El.innerText = e20.toFixed(5);
        }}

        setLastBarLegend();

        chart.subscribeCrosshairMove(param => {{
            if (!param.time || !param.seriesData) {{
                setLastBarLegend();
                return;
            }}
            const bar = param.seriesData.get(candleSeries);
            const e5 = param.seriesData.get(ema5Series);
            const e20 = param.seriesData.get(ema20Series);
            updateLegend(bar, e5?.value, e20?.value);
        }});

        chart.timeScale().fitContent();

        const ro = new ResizeObserver(entries => {{
            if (entries.length && entries[0].contentRect) {{
                chart.applyOptions({{ width: entries[0].contentRect.width }});
            }}
        }});
        ro.observe(container);
    </script>
</body>
</html>"""
    return html

