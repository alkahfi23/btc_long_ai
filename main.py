import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime
import requests
import time

st.set_page_config(page_title="AI Crypto Signal Analyzer", layout="wide")
st.title("📊 AI Crypto Signal Analyzer (Real-Time AscendEX Futures)")
st.sidebar.title("🔧 Pengaturan Analisa")

# Gunakan hanya dua pair untuk AscendEX Futures
usdt_pairs = ["BTCUSDT", "ETHUSDT"]

selected_pair = st.sidebar.selectbox("💱 Pilih Pair", usdt_pairs, index=usdt_pairs.index("BTCUSDT"))
timeframe = st.sidebar.selectbox("⏱️ Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=0)
modal = st.sidebar.number_input("💰 Modal ($)", value=1000.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", 0.1, 5.0, 1.0)
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
margin_mode = st.sidebar.radio("💼 Mode Margin", ["Cross", "Isolated"], index=0)
start_analysis = st.sidebar.button("🚀 Mulai Analisa")
refresh_data = st.sidebar.button("🔄 Refresh Manual")

if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = 0

@st.cache_data(ttl=60)
def fetch_initial_data(symbol, interval, limit=200):
    try:
        interval_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60"
        }
        resolution = interval_map.get(interval, "1")
        url = f"https://ascendex.com/api/pro/v1/futures/marketdata/candlestick?symbol={symbol.lower()}/perp&interval={resolution}&n={limit}"
        response = requests.get(url)
        data = response.json()
        candles = data.get("data", {}).get("bars", [])
        df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(c[0], unit="s"),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4])
            }
            for c in candles
        ])
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data historis: {e}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

current_time = time.time()
if current_time - st.session_state.last_fetch_time > 15 or refresh_data:
    st.session_state.price_data = fetch_initial_data(selected_pair, timeframe)
    st.session_state.last_fetch_time = current_time

if start_analysis:
    st.success("✅ Data diambil, chart dan analisa ditampilkan di bawah.")

# Tampilkan chart dengan indikator
if len(st.session_state.price_data) >= 20:
    df = st.session_state.price_data.copy()
    df.set_index("timestamp", inplace=True)
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df.dropna(inplace=True)

    signal = ""
    entry_price = df["close"].iloc[-1]
    atr = df["close"].rolling(window=14).std().iloc[-1] * 1.5
    stop_loss = None
    take_profit = None

    if (
        df["rsi"].iloc[-1] < 30 and
        df["close"].iloc[-1] > df["ema_fast"].iloc[-1] and
        df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
    ):
        signal = "📈 LONG"
        stop_loss = entry_price - atr
        take_profit = entry_price + (2 * atr)
    elif (
        df["rsi"].iloc[-1] > 70 and
        df["close"].iloc[-1] < df["ema_fast"].iloc[-1] and
        df["macd"].iloc[-1] < df["macd_signal"].iloc[-1]
    ):
        signal = "📉 SHORT"
        stop_loss = entry_price + atr
        take_profit = entry_price - (2 * atr)

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candlestick"
        ),
        go.Scatter(x=df.index, y=df["ema_fast"], mode="lines", name="EMA 5"),
        go.Scatter(x=df.index, y=df["ema_slow"], mode="lines", name="EMA 20")
    ])

    if signal:
        fig.add_hline(y=entry_price, line=dict(color="blue", dash="dash"), annotation_text=f"Entry: {entry_price:.2f}", annotation_position="top left")
        fig.add_hline(y=stop_loss, line=dict(color="red", dash="dot"), annotation_text=f"SL: {stop_loss:.2f}", annotation_position="bottom left")
        fig.add_hline(y=take_profit, line=dict(color="green", dash="dot"), annotation_text=f"TP: {take_profit:.2f}", annotation_position="top right")

    fig.update_layout(title=f"📈 Real-Time Chart: {selected_pair}", xaxis_title="Waktu", yaxis_title="Harga")
    st.plotly_chart(fig, use_container_width=True)

    if signal:
        st.success(f"🔔 Sinyal: {signal}\n💵 Entry: {entry_price:.2f}\n🛑 SL: {stop_loss:.2f}\n🎯 TP: {take_profit:.2f}")
    else:
        st.info("🔍 Belum ada sinyal valid")
else:
    st.info("📡 Klik tombol 'Mulai Analisa' untuk memulai polling data dari AscendEX")
