import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import websocket
import json
import threading
from datetime import datetime
import requests
import time
import contextlib

st.set_page_config(page_title="AI Crypto Signal Analyzer", layout="wide")
st.title("📊 AI Crypto Signal Analyzer (Real-Time Binance)")
st.sidebar.title("🔧 Pengaturan Analisa")

@st.cache_data(ttl=600)
def get_binance_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
        return sorted(symbols)
    except Exception as e:
        st.error(f"Gagal mengambil daftar pair: {e}")
        return ["BTCUSDT", "ETHUSDT"]

usdt_pairs = get_binance_usdt_pairs()
selected_pair = st.sidebar.selectbox("💱 Pilih Pair", usdt_pairs, index=usdt_pairs.index("BTCUSDT"))
timeframe = st.sidebar.selectbox("⏱️ Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=0)
modal = st.sidebar.number_input("💰 Modal ($)", value=1000.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", 0.1, 5.0, 1.0)
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
margin_mode = st.sidebar.radio("💼 Mode Margin", ["Cross", "Isolated"], index=0)
start_analysis = st.sidebar.button("🚀 Mulai Analisa")

placeholder = st.empty()
price_data = pd.DataFrame(columns=["timestamp", "price"])

@st.cache_data(ttl=60)
def fetch_initial_data(symbol, interval, limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        klines = response.json()
        data = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(k[0], unit="ms"),
                "price": float(k[4])
            }
            for k in klines
        ])
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data historis: {e}")
        return pd.DataFrame(columns=["timestamp", "price"])

def on_message(ws, message):
    global price_data
    data = json.loads(message)
    price = float(data['k']['c'])
    timestamp = pd.to_datetime(data['k']['t'], unit='ms')
    new_row = pd.DataFrame({"timestamp": [timestamp], "price": [price]})
    price_data = pd.concat([price_data, new_row], ignore_index=True)
    price_data = price_data.drop_duplicates(subset="timestamp", keep="last")
    price_data = price_data.tail(200)

    if len(price_data) >= 35:
        price_data.set_index("timestamp", inplace=True)
        df = price_data.copy()
        df["ema_fast"] = ta.trend.EMAIndicator(df["price"], window=5).ema_indicator()
        df["ema_slow"] = ta.trend.EMAIndicator(df["price"], window=20).ema_indicator()
        df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["volume"] = df["price"].diff().abs()
        df["volume_avg"] = df["volume"].rolling(window=20).mean()
        df.dropna(inplace=True)

        signal = ""
        entry_price = df["price"].iloc[-1]
        atr = df["price"].rolling(window=14).std().iloc[-1] * 1.5
        stop_loss = take_profit = None

        rsi = df["rsi"].iloc[-1]
        close = df["price"].iloc[-1]
        ema_fast = df["ema_fast"].iloc[-1]
        ema_slow = df["ema_slow"].iloc[-1]
        macd_line = df["macd"].iloc[-1]
        macd_signal = df["macd_signal"].iloc[-1]
        vol_now = df["volume"].iloc[-1]
        vol_avg = df["volume_avg"].iloc[-1]

        if (
            rsi < 30 and close > ema_fast and close > ema_slow and
            macd_line > macd_signal and vol_now > vol_avg
        ):
            signal = "📈 LONG"
            stop_loss = entry_price - atr
            take_profit = entry_price + 2 * atr

        elif (
            rsi > 70 and close < ema_fast and close < ema_slow and
            macd_line < macd_signal and vol_now > vol_avg
        ):
            signal = "📉 SHORT"
            stop_loss = entry_price + atr
            take_profit = entry_price - 2 * atr

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["price"], mode="lines", name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], mode="lines", name="EMA 5"))
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], mode="lines", name="EMA 20"))

        if signal:
            fig.add_hline(y=entry_price, line=dict(color="blue", dash="dash"), annotation_text=f"Entry: {entry_price:.2f}", annotation_position="top left")
            fig.add_hline(y=stop_loss, line=dict(color="red", dash="dot"), annotation_text=f"SL: {stop_loss:.2f}", annotation_position="bottom left")
            fig.add_hline(y=take_profit, line=dict(color="green", dash="dot"), annotation_text=f"TP: {take_profit:.2f}", annotation_position="top right")

        fig.update_layout(title=f"📈 Real-Time Chart: {selected_pair}", xaxis_title="Waktu", yaxis_title="Harga")
        placeholder.plotly_chart(fig, use_container_width=True)

        if signal:
            st.success(f"🔔 Sinyal: {signal}\n💵 Entry: {entry_price:.2f}\n🚩 SL: {stop_loss:.2f}\n🌟 TP: {take_profit:.2f}")
        else:
            st.info("🔍 Belum ada sinyal valid")

    price_data.reset_index(drop=True, inplace=True)

def on_error(ws, error):
    st.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket ditutup")

def run_ws():
    stream = f"wss://stream.binance.com:9443/ws/{selected_pair.lower()}@kline_{timeframe}"
    ws = websocket.WebSocketApp(stream, on_message=on_message, on_error=on_error, on_close=on_close)
    with contextlib.suppress(Exception):
        ws.run_forever()

if start_analysis:
    with st.spinner("⏳ Mengambil data historis..."):
        price_data = fetch_initial_data(selected_pair, timeframe)

    threading.Thread(target=run_ws, daemon=True).start()
    st.success("✅ Analisa dimulai, data real-time sedang berjalan...")
else:
    st.info("📁 Klik tombol 'Mulai Analisa' untuk memulai streaming data dari Binance WebSocket")
