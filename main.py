import streamlit as st
import pandas as pd
import numpy as np
import websockets
import asyncio
import json
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.title("📊 AI BTC/ETH Signal Analyzer")
st.sidebar.title("🔧 Pengaturan Analisa")

# Fungsi untuk mendapatkan data candle melalui WebSocket Binance
async def get_kline_data(symbol, interval="1m"):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            kline = data['k']['c']  # 'c' adalah harga penutupan
            timestamp = data['k']['T']
            yield pd.DataFrame({
                'timestamp': [pd.to_datetime(timestamp, unit='ms')],
                'close': [float(kline)],
                'open': [float(data['k']['o'])],
                'high': [float(data['k']['h'])],
                'low': [float(data['k']['l'])],
                'volume': [float(data['k']['v'])]
            })

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
    atr_buffer = df["atr"] * 1.5

    df.loc[
        (df["rsi"] < 30) &
        (df["rsi"] > 10) &
        (df["macd"] > df["macd_signal"]) &
        (df["close"] > df["ema_fast"]) &
        (df["ema_fast"] > df["ema_slow"]) &
        (df["close"] < df["bb_lower"]) &
        (df["adx"] > adx_threshold) &
        ((df["ema_fast"] - df["ema_slow"]) > df["atr"] * 0.2) &
        ((df["close"] - df["ema_fast"]) > df["atr"] * 0.2),
        "signal"
    ] = "LONG"

    df.loc[
        (df["rsi"] > 70) &
        (df["rsi"] < 90) &
        (df["macd"] < df["macd_signal"]) &
        (df["close"] < df["ema_fast"]) &
        (df["ema_fast"] < df["ema_slow"]) &
        (df["close"] > df["bb_upper"]) &
        (df["adx"] > adx_threshold) &
        ((df["ema_slow"] - df["ema_fast"]) > df["atr"] * 0.2) &
        ((df["ema_fast"] - df["close"]) > df["atr"] * 0.2),
        "signal"
    ] = "SHORT"

    return df

# Fungsi untuk memperbarui tampilan grafik
def update_chart(df, fig):
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df["ema_fast"], line=dict(color='blue', width=1), name="EMA 5"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df["ema_slow"], line=dict(color='orange', width=1), name="EMA 20"))
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk menampilkan sinyal
def display_signals(df):
    recent_signals = df[df["signal"].isin(["LONG", "SHORT"])].copy()
    if not recent_signals.empty:
        last_row = recent_signals.iloc[-1]
        last_signal = last_row["signal"]
        entry = last_row["close"]
        atr = last_row["atr"]

        if last_signal == "LONG":
            sl, tp = calculate_sl_tp(entry, atr, is_long=True)
        else:
            sl, tp = calculate_sl_tp(entry, atr, is_long=False)

        st.write(f"**Sinyal:** {last_signal}")
        st.write(f"**Harga Entry:** ${entry:.2f}")
        st.write(f"**Stop Loss:** ${sl:.2f}")
        st.write(f"**Take Profit:** ${tp:.2f}")

        # Manajemen Risiko
        risk_amount = modal * (risk_pct / 100)
        position_size = (risk_amount * leverage) / abs(entry - sl)
        position_value = position_size * entry

        st.markdown("### 💼 Manajemen Risiko")
        st.write(f"**Risiko Maksimal:** ${risk_amount:.2f}")
        st.write(f"**Ukuran Posisi (leverage {leverage}x):** {position_size:.4f} {symbol}")
        st.write(f"**Nilai Posisi:** ${position_value:.2f}")

        # Estimasi Profit & Loss
        potential_profit = abs(tp - entry) * position_size
        potential_loss = abs(entry - sl) * position_size

        st.markdown("### 📈 Estimasi Potensi Profit & Loss")
        st.write(f"**Potensi Profit:** ${potential_profit:.2f}")
        st.write(f"**Potensi Loss:** ${potential_loss:.2f}")
    else:
        st.info("Belum ada sinyal LONG/SHORT yang valid dalam data terbaru.")

# Fungsi utama untuk Streamlit yang menjalankan WebSocket dan pembaruan data
if st.sidebar.button("🚀 Jalankan Analisa"):
    symbol = st.sidebar.selectbox("💱 Pilih Pair Trading", options=["btcusdt", "ethusdt"], index=0)
    interval = st.sidebar.selectbox("⏱️ Pilih Timeframe Trading", options=["1m", "5m", "15m", "30m", "1h"], index=0)

    # Create a plotly chart
    fig = go.Figure()

    # Start the WebSocket
    async def main():
        async for new_data in get_kline_data(symbol, interval):
            # Update data and chart in real-time
            df = new_data
            df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
            df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
            df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()
            df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            bb = ta.volatility.BollingerBands(df["close"])
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()

            df = get_valid_signal(df)
            display_signals(df)
            update_chart(df, fig)

    asyncio.run(main())
