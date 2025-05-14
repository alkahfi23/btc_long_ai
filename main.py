import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.sidebar.title("🔧 Pengaturan Analisa")

# Sidebar inputs
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0)
interval = st.sidebar.selectbox("⏱️ Pilih Interval Waktu", options=["1", "5", "15", "30", "60", "240", "D"], index=4)
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
mode_strategi = st.sidebar.selectbox("📈 Mode Strategi", ["Adaptif", "Agresif", "Konservatif"])
mode_data = st.sidebar.selectbox("🔄 Mode Data", ["Live", "Backtest"])
symbol = st.sidebar.selectbox("💱 Simbol", options=["BTC", "ETH"], index=0)

uploaded_file = None
df_csv = None
if mode_data == "Backtest":
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Harga BTC/ETH", type=["csv"])
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        df_csv.set_index("timestamp", inplace=True)
        df_csv.sort_index(inplace=True)

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = []

@st.cache_data(ttl=300)
def get_kline_data(symbol, interval="60", limit=200, csv_df=None):
    if csv_df is not None:
        df = csv_df.copy()
    else:
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol + "USDT", "interval": interval, "limit": limit}
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            df = pd.DataFrame(data["result"]["list"])
            df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
            numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df.set_index("timestamp", inplace=True)
        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    df.sort_index(inplace=True)
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

# Fungsi untuk mendapatkan sinyal valid (berdasarkan indikator teknikal)
def get_valid_signal(df):
    df["signal"] = ""
    
    # Tentukan sinyal berdasarkan indikator teknikal
    df.loc[
        (df["rsi"] > 70) & (df["close"] > df["ema_fast"]) & (df["macd"] > 0), "signal"
    ] = "SHORT"
    
    df.loc[
        (df["rsi"] < 30) & (df["close"] < df["ema_fast"]) & (df["macd"] < 0), "signal"
    ] = "LONG"

    # Ambil sinyal yang valid
    valid_signals = df[df["signal"] != ""]
    return valid_signals

# Chart dengan Entry, Take Profit, Stop Loss
def plot_chart(df, entry, stop_loss, take_profit):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Harga Aktual"))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[entry], mode="markers", name="Entry", marker=dict(size=10, color="blue")))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[take_profit], mode="markers", name="Take Profit", marker=dict(size=10, color="green")))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[stop_loss], mode="markers", name="Stop Loss", marker=dict(size=10, color="red")))
    fig.update_layout(title="📈 Chart Harga, Entry, TP, SL", xaxis_title="Waktu", yaxis_title="Harga USD")
    st.plotly_chart(fig, use_container_width=True)

# Menampilkan sinyal untuk BTC dan ETH tanpa filter
def display_signals(symbol):
    df = get_kline_data(symbol)
    if not df.empty:
        valid_signals = get_valid_signal(df)
        if not valid_signals.empty:
            for idx, row in valid_signals.iterrows():
                entry = row["close"]
                stop_loss = entry * 0.98 if row["signal"] == "LONG" else entry * 1.02
                take_profit = entry * 1.03 if row["signal"] == "LONG" else entry * 0.97
                st.write(f"**{symbol}** - Signal {row['signal']} at {entry:.2f}")
                plot_chart(df, entry, stop_loss, take_profit)
        else:
            st.write(f"Tidak ada sinyal valid untuk {symbol}")
    else:
        st.write(f"Gagal mengambil data untuk {symbol}")

# Menampilkan sinyal untuk BTC dan ETH
display_signals("BTC")
display_signals("ETH")
