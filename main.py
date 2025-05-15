import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime
import requests
import threading
import time
import json
from websocket import WebSocketApp
import os

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

latest_data = []

def handle_socket_message(ws, msg):
    try:
        data = json.loads(msg)
        if data.get("m") == "bar":
            market_data = data.get("data", {})
            candle = {
                "timestamp": pd.to_datetime(int(market_data["ts"]) / 1000, unit='s'),
                "open": float(market_data["o"]),
                "high": float(market_data["h"]),
                "low": float(market_data["l"]),
                "close": float(market_data["c"])
            }
            latest_data.append(candle)
            st.session_state.last_price = float(market_data["c"])
    except Exception as e:
        print(f"WebSocket message error: {e}")

def start_ascendex_futures_socket():
    def on_message(ws, message):
        handle_socket_message(ws, message)

    def on_error(ws, error):
        print(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        if close_status_code == 1000:
            print("Normal closure. Reconnecting...")
            time.sleep(2)
            run()  # restart

    def on_open(ws):
        try:
            interval_map = {
                "1m": "1",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60"
            }
            resolution = interval_map.get(timeframe, "1")
            symbol = selected_pair.lower()
            channel = f"bar:{symbol}/perp:{resolution}"
            subscribe_payload = {
                "op": "sub",
                "ch": channel,
                "id": f"sub-{int(time.time())}"
            }
            ws.send(json.dumps(subscribe_payload))
            print(f"✅ Subscribed to {channel}")
        except Exception as e:
            print(f"Failed to subscribe: {e}")

    def run():
        while True:
            try:
                ws_app = WebSocketApp(
                    "wss://ascendex.com/1/api/pro/v1/stream",
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                ws_app.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"WebSocket connection failed, retrying... {e}")
                time.sleep(5)

    threading.Thread(target=run, daemon=True).start()

if start_analysis and 'ws_thread' not in st.session_state:
    with st.spinner("⏳ Mengambil data historis..."):
        st.session_state.price_data = fetch_initial_data(selected_pair, timeframe)

    start_ascendex_futures_socket()
    st.session_state.ws_thread = True
    st.success("✅ Analisa dimulai, data real-time sedang berjalan...")


# (bagian analisis dan visualisasi tetap sama, tidak perlu diubah)

if latest_data:
    new_rows = pd.DataFrame(latest_data)
    st.session_state.price_data = pd.concat([st.session_state.price_data, new_rows], ignore_index=True)
    st.session_state.price_data.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    st.session_state.price_data = st.session_state.price_data.tail(200)
    latest_data.clear()

last_price = st.session_state.get("last_price", None)
if last_price:
    st.metric(label="📈 Harga Terakhir (Real-Time)", value=f"{last_price:.2f} USDT")

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
    atr_buffer = atr * 1.2
    stop_loss = None
    take_profit = None

    rsi_now = df["rsi"].iloc[-1]
    rsi_prev = df["rsi"].iloc[-2]
    price_now = df["close"].iloc[-1]
    ema_fast = df["ema_fast"].iloc[-1]
    ema_slow = df["ema_slow"].iloc[-1]
    macd_now = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]

    if (
        rsi_now > 40 and rsi_now > rsi_prev and
        price_now > ema_fast and ema_fast > ema_slow and
        macd_now > macd_signal
    ):
        signal = "📈 LONG"
        stop_loss = entry_price - atr_buffer
        take_profit = entry_price + (2.5 * atr)
    elif (
        rsi_now < 60 and rsi_now < rsi_prev and
        price_now < ema_fast and ema_fast < ema_slow and
        macd_now < macd_signal
    ):
        signal = "📉 SHORT"
        stop_loss = entry_price + atr_buffer
        take_profit = entry_price - (2.5 * atr)

    risk_amount = modal * (risk_pct / 100)
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit > 0:
        position_size = risk_amount / risk_per_unit
        position_value = position_size * entry_price
        contract_qty = (position_value * leverage) / entry_price
    else:
        position_size = 0
        contract_qty = 0

    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=("Candlestick + EMA", "RSI & MACD"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candlestick"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], mode="lines", name="EMA 5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], mode="lines", name="EMA 20"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], mode="lines", name="MACD Signal"), row=2, col=1)

    if signal:
        fig.add_hline(y=entry_price, line=dict(color="blue", dash="dash"), row=1, col=1,
                      annotation_text=f"Entry: {entry_price:.2f}", annotation_position="top left")
        fig.add_hline(y=stop_loss, line=dict(color="red", dash="dot"), row=1, col=1,
                      annotation_text=f"SL: {stop_loss:.2f}", annotation_position="bottom left")
        fig.add_hline(y=take_profit, line=dict(color="green", dash="dot"), row=1, col=1,
                      annotation_text=f"TP: {take_profit:.2f}", annotation_position="top right")

    fig.update_layout(title=f"📈 Real-Time Chart: {selected_pair}", xaxis_title="Waktu", yaxis_title="Harga",
                      height=700)
    st.plotly_chart(fig, use_container_width=True)

    if signal:
        st.success(
            f"🔔 Sinyal: {signal}\n"
            f"💵 Entry: {entry_price:.2f} USDT\n"
            f"🛑 SL: {stop_loss:.2f} | 🎯 TP: {take_profit:.2f}\n"
            f"📐 Risk per Unit: {risk_per_unit:.2f} | 💸 Risk Amount: ${risk_amount:.2f}\n"
            f"📊 Position Size: {position_size:.4f} | 📦 Qty Kontrak (@{leverage}x): {contract_qty:.2f}"
        )
    else:
        st.info("🔍 Belum ada sinyal valid")
else:
    st.info("📡 Klik tombol 'Mulai Analisa' untuk memulai streaming data dari AscendEX Futures WebSocket")
