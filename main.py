import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import io

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.sidebar.title("🔧 Pengaturan Analisa")

# Sidebar inputs
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0)
interval = st.sidebar.selectbox("⏱️ Pilih Interval Waktu", options=["1", "5", "15", "30", "60", "240", "D"], index=4)
signal_filter = st.sidebar.selectbox("🧳 Filter Sinyal", options=["SEMUA", "LONG", "SHORT", "WAIT"])
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
mode_strategi = st.sidebar.selectbox("📈 Mode Strategi", ["Adaptif", "Agresif", "Konservatif"])

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = []

@st.cache_data(ttl=300)
def get_kline_data(symbol, interval="60", limit=200):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        df = pd.DataFrame(data["result"]["list"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # Tambahan indikator
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
        df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["macd"] = ta.trend.MACD(df["close"]).macd()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()

        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def detect_market_regime(df):
    adx = df["adx"]
    ema_fast = df["ema_fast"]
    ema_slow = df["ema_slow"]
    trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1] and adx.iloc[-1] > 20
    trend_down = ema_fast.iloc[-1] < ema_slow.iloc[-1] and adx.iloc[-1] > 20
    if trend_up:
        return "UPTREND"
    elif trend_down:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

def detect_signal(df):
    last = df.iloc[-1]
    signal, tp, sl = "WAIT", None, None
    regime = detect_market_regime(df)

    if regime == "UPTREND":
        if (
            last["ema_fast"] > last["ema_slow"] and
            last["rsi"] > 50 and
            last["macd"] > 0 and
            last["stoch_k"] > last["stoch_d"]
        ):
            signal = "LONG"
            tp = last["close"] + 2 * last["atr"]
            sl = last["close"] - 1.5 * last["atr"]

    elif regime == "DOWNTREND":
        if (
            last["ema_fast"] < last["ema_slow"] and
            last["rsi"] < 50 and
            last["macd"] < 0 and
            last["stoch_k"] < last["stoch_d"]
        ):
            signal = "SHORT"
            tp = last["close"] - 2 * last["atr"]
            sl = last["close"] + 1.5 * last["atr"]

    elif regime == "SIDEWAYS":
        if (
            last["close"] < last["bb_lower"] and
            last["stoch_k"] < 20 and
            last["rsi"] < 30
        ):
            signal = "LONG"
            tp = last["close"] + 2 * last["atr"]
            sl = last["close"] - 1.5 * last["atr"]
        elif (
            last["close"] > last["bb_upper"] and
            last["stoch_k"] > 80 and
            last["rsi"] > 70
        ):
            signal = "SHORT"
            tp = last["close"] - 2 * last["atr"]
            sl = last["close"] + 1.5 * last["atr"]

    entry = last["close"]
    return signal, entry, tp, sl

def plot_chart(df, signal):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], line=dict(color='blue', width=1), name='EMA 5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], line=dict(color='orange', width=1), name='EMA 20'))
    fig.update_layout(title=f"BTCUSDT - Sinyal: {signal}", xaxis_title='Waktu', yaxis_title='Harga')
    return fig

# Evaluasi dan tampilkan sinyal hanya untuk BTCUSDT dan ETHUSDT
for symbol in ["BTCUSDT", "ETHUSDT"]:
    df = get_kline_data(symbol, interval=interval)
    if not df.empty:
        signal, entry, tp, sl = detect_signal(df)
        if signal_filter == "SEMUA" or signal == signal_filter:
            st.subheader(f"📊 {symbol}")
            chart = plot_chart(df, signal)
            st.plotly_chart(chart, use_container_width=True)
            st.metric("Sinyal", signal)
            if signal != "WAIT":
                st.write(f"Entry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
