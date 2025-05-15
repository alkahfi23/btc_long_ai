# main.py
import streamlit as st
import pandas as pd
import time
import requests
import ta
import plotly.graph_objects as go

st.set_page_config(page_title="AI Crypto Signal Analyzer (Bybit Futures)", layout="wide")
st.title("📊 AI Crypto Signal Analyzer (Real-Time Bybit Futures)")
st.sidebar.title("🔧 Pengaturan Analisa")

usdt_pairs = ["BTCUSDT", "ETHUSDT"]
selected_pair = st.sidebar.selectbox("💱 Pilih Pair", usdt_pairs, index=0)
timeframe = st.sidebar.selectbox("⏱️ Timeframe", ["1", "3", "5", "15", "30", "60"], index=0)  # sesuai Bybit API intervals
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

def fetch_bybit_ohlcv(symbol, interval, limit=200):
    url = "https://api.bybit.com/public/linear/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("ret_code") != 0:
            st.error(f"API Error: {data.get('ret_msg', 'Unknown error')}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        klines = data.get("result", [])
        if not klines:
            st.error("Data klines kosong dari Bybit API")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        df = pd.DataFrame(klines)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="s")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        return df[["timestamp", "open", "high", "low", "close"]]
    except Exception as e:
        st.error(f"Gagal ambil data OHLCV: {e}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

current_time = time.time()
if current_time - st.session_state.last_fetch_time > 15 or refresh_data:
    st.session_state.price_data = fetch_bybit_ohlcv(selected_pair, timeframe)
    st.session_state.last_fetch_time = current_time

if start_analysis:
    st.success("✅ Data diambil, chart dan analisa ditampilkan di bawah.")

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
    st.info("📡 Klik tombol 'Mulai Analisa' untuk memulai polling data dari Bybit")
