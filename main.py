import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from binance.client import Client  # Import client untuk autentikasi
from binance.websockets import BinanceSocketManager  # WebSocket Manager untuk futures
from datetime import datetime
import requests
import threading
import time

# Konfigurasi Streamlit
st.set_page_config(page_title="AI Crypto Signal Analyzer", layout="wide")
st.title("📊 AI Crypto Signal Analyzer (Real-Time Binance Futures)")
st.sidebar.title("🔧 Pengaturan Analisa")

# Fungsi untuk mengambil semua pair USDT dari Binance
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
refresh_data = st.sidebar.button("🔄 Refresh Manual")

# Inisialisasi DataFrame untuk harga
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

# Ambil data historis
@st.cache_data(ttl=60)
def fetch_initial_data(symbol, interval, limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        klines = response.json()
        data = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4])
            }
            for k in klines
        ])
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data historis: {e}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

latest_data = []

def handle_socket_message(msg):
    if msg.get("k"):
        k = msg["k"]
        candle = {
            "timestamp": pd.to_datetime(k['t'], unit='ms'),
            "open": float(k['o']),
            "high": float(k['h']),
            "low": float(k['l']),
            "close": float(k['c'])
        }
        latest_data.append(candle)

def start_binance_socket():
    # Inisialisasi koneksi WebSocket untuk Binance Futures
    client = Client(api_key='1a6IOutOrQAauWOz3r0RsZYspF4zujeM8QB9OBRb5tPB6wZ2GRaEF09EX1WjihlH', api_secret='3HpKjPR26l5Gy6cjOKpLqUCnEcidqpqfTwgN1pV6upzt6XKIbYYjRoGmV1pwfn9Q')
    bm = BinanceSocketManager(client)
    conn_key = bm.start_kline_socket(selected_pair.lower(), timeframe, handle_socket_message)
    bm.start()

    while True:
        time.sleep(1)  # Keep thread alive

if start_analysis and 'ws_thread' not in st.session_state:
    with st.spinner("⏳ Mengambil data historis..."):
        st.session_state.price_data = fetch_initial_data(selected_pair, timeframe)

    ws_thread = threading.Thread(target=start_binance_socket, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread
    st.success("✅ Analisa dimulai, data real-time sedang berjalan...")

# Menambahkan data terbaru dari latest_data ke session_state
if latest_data:
    new_rows = pd.DataFrame(latest_data)
    st.session_state.price_data = pd.concat([st.session_state.price_data, new_rows], ignore_index=True)
    st.session_state.price_data.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    st.session_state.price_data = st.session_state.price_data.tail(200)
    latest_data.clear()

# Hitung indikator dan tampilkan chart
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
    st.info("📡 Klik tombol 'Mulai Analisa' untuk memulai streaming data dari Binance WebSocket")
