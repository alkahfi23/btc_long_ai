import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.title("📊 AI BTC/ETH Signal Analyzer")
st.sidebar.title("🔧 Pengaturan Analisa")

# Fungsi untuk mengambil daftar trading pair dari API Bybit (v5)
def get_trading_pairs():
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}
    try:
        res = requests.get(url, params=params, timeout=10)

        if res.status_code != 200:
            st.error(f"Gagal mendapatkan data dari Bybit (status {res.status_code})")
            return []

        try:
            data = res.json()
        except ValueError:
            st.error("Respon dari API bukan format JSON.")
            return []

        if "result" in data and "list" in data["result"]:
            pairs = [item['symbol'] for item in data['result']['list'] if item['symbol'].endswith('USDT')]
            if not pairs:
                st.warning("Tidak ada pair USDT ditemukan.")
            return pairs
        else:
            st.error("Struktur data API tidak sesuai.")
            return []

    except Exception as e:
        st.error(f"Error fetching trading pairs: {e}")
        return []

# Ambil daftar trading pairs dari Bybit
trading_pairs = get_trading_pairs()

# Sidebar inputs untuk pengaturan analisis
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0)
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
margin_mode = st.sidebar.radio("💼 Mode Margin", options=["Cross", "Isolated"], index=0)

symbol = st.sidebar.selectbox("💱 Pilih Pair Trading", options=trading_pairs, index=0)
timeframe = st.sidebar.selectbox("⏱️ Pilih Timeframe Trading", options=["1", "5", "15", "30", "60", "240", "D"], index=4)

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
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

    # Menambahkan indikator teknikal
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    return df

# Fungsi untuk menghitung SL dan TP berdasarkan ATR
def calculate_sl_tp(entry, atr, is_long=True, sl_multiplier=2, tp_multiplier=3):
    if is_long:
        stop_loss = entry - atr * sl_multiplier
        take_profit = entry + atr * tp_multiplier
    else:
        stop_loss = entry + atr * sl_multiplier
        take_profit = entry - atr * tp_multiplier
    return stop_loss, take_profit

# Fungsi untuk mendapatkan sinyal valid
def get_valid_signal(df):
    df["signal"] = ""
    adx_threshold = 20

    df.loc[
        (df["rsi"] < 30) &
        (df["macd"] > 0) &
        (df["close"] > df["ema_fast"]) &
        (df["adx"] > adx_threshold),
        "signal"
    ] = "LONG"

    df.loc[
        (df["rsi"] > 70) &
        (df["macd"] < 0) &
        (df["close"] < df["ema_fast"]) &
        (df["adx"] > adx_threshold),
        "signal"
    ] = "SHORT"

    return df

# Tombol untuk memulai analisis
if st.sidebar.button("🚀 Jalankan Analisa"):
    df = get_kline_data(symbol, interval=timeframe)
    if not df.empty:
        df = get_valid_signal(df)
        last_signal = df["signal"].iloc[-1]
        atr = df["atr"].iloc[-1]
        entry = df["close"].iloc[-1]

        if last_signal == "LONG":
            sl, tp = calculate_sl_tp(entry, atr, is_long=True)
        elif last_signal == "SHORT":
            sl, tp = calculate_sl_tp(entry, atr, is_long=False)
        else:
            sl, tp = None, None

        st.subheader(f"📈 Sinyal Terbaru untuk {symbol} ({timeframe}m)")
        st.write(f"**Sinyal:** {last_signal}")
        st.write(f"**Harga Entry:** ${entry:.2f}")
        if sl and tp:
            st.write(f"**Stop Loss:** ${sl:.2f}")
            st.write(f"**Take Profit:** ${tp:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candles'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], line=dict(color='blue', width=1), name="EMA 5"))
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], line=dict(color='orange', width=1), name="EMA 20"))
        st.plotly_chart(fig, use_container_width=True)

        # Visualisasi RSI
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="purple")))
        rsi_fig.add_hline(y=70, line=dict(color="red", dash="dash"))
        rsi_fig.add_hline(y=30, line=dict(color="green", dash="dash"))
        rsi_fig.update_layout(title="RSI", height=300)
        st.plotly_chart(rsi_fig, use_container_width=True)

        # Visualisasi MACD dan Histogram
        histogram = df["macd"] - df["macd_signal"]
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="blue")))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="orange")))
        macd_fig.add_trace(go.Bar(x=df.index, y=histogram, name="Histogram", marker_color="gray"))
        macd_fig.update_layout(title="MACD", height=300)
        st.plotly_chart(macd_fig, use_container_width=True)

        # Manajemen Risiko
        if sl:
            risk_amount = modal * (risk_pct / 100)
            position_size = (risk_amount * leverage) / abs(entry - sl)
            position_value = position_size * entry

            st.markdown("### 💼 Manajemen Risiko")
            st.write(f"**Risiko Maksimal:** ${risk_amount:.2f}")
            st.write(f"**Ukuran Posisi (leverage {leverage}x):** {position_size:.4f} {symbol}")
            st.write(f"**Nilai Posisi:** ${position_value:.2f}")

        st.session_state.analyzed = True
    else:
        st.warning("Data tidak tersedia atau gagal memuat.")
