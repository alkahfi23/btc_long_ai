# ========== IMPORT ========== #
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import ta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# ========== STREAMLIT CONFIG ========== #
st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.title("🤖 AI BTC/ETH Signal Analyzer (with LSTM & Chart)")

# ========== API HISTORICAL DATA BYBIT ========== #
@st.cache_data(ttl=600)
def get_bybit_data(symbol="BTCUSDT", interval="60"):
    dfs = []
    limit = 200  # Max per API call
    now = int(datetime.datetime.now().timestamp() * 1000)
    ms_per_candle = int(interval) * 60 * 1000
    lookback = 365 * 24 * 60 // int(interval)

    for i in range(0, lookback, limit):
        start = now - (i + limit) * ms_per_candle
        end = now - i * ms_per_candle
        url = f"https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "start": start,
            "end": end
        }
        try:
            r = requests.get(url, params=params)
            if r.status_code == 200 and "result" in r.json():
                data = r.json()["result"]["list"]
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df = df.astype(float)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                dfs.append(df)
        except Exception as e:
            print(f"Bybit API error: {e}")
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()

# ========== API HISTORICAL DATA BINANCE (fallback) ========== #
@st.cache_data(ttl=600)
def get_binance_data(symbol="BTCUSDT", interval="60"):
    url = f"https://api.binance.com/api/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 200  # Max per API call
    }
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_vol", "number_of_trades", "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"])
            df = df.astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
    except Exception as e:
        print(f"Binance API error: {e}")
    return pd.DataFrame()

# ========== INDICATORS ========== #
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    ichimoku = ta.trend.IchimokuIndicator(df["high"], df["low"])
    df["ichimoku_a"] = ichimoku.ichimoku_a()
    df["ichimoku_b"] = ichimoku.ichimoku_b()
    return df

# ========== LSTM MODEL ========== #
def train_lstm_model(df, n_steps=60):
    data = df[['close']].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i-n_steps:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    return model, scaler

def predict_future_price(df, model, scaler, n_steps=60):
    last_seq = df[['close']].dropna().iloc[-n_steps:]
    scaled_input = scaler.transform(last_seq)
    input_reshaped = scaled_input.reshape((1, n_steps, 1))
    pred_scaled = model.predict(input_reshaped)[0][0]
    return scaler.inverse_transform([[pred_scaled]])[0][0]

# ========== FIBONACCI ========== #
def fibonacci_levels(df):
    high, low = df["high"].max(), df["low"].min()
    diff = high - low
    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100.0%": low
    }

# ========== SINYAL ========== #
def detect_signal(df):
    last = df.iloc[-1]
    price = last["close"]
    atr = last["atr"]
    long_cond = last["rsi"] < 30 and last["close"] < last["bb_low"] and last["ema_fast"] > last["ema_slow"]
    short_cond = last["rsi"] > 70 and last["close"] > last["bb_high"] and last["ema_fast"] < last["ema_slow"]

    if long_cond:
        return "LONG", price, price + 2*atr, price - 1.5*atr
    elif short_cond:
        return "SHORT", price, price - 2*atr, price + 1.5*atr
    else:
        return "WAIT", None, None, None

# ========== VISUALISASI ========== #
def plot_chart(df, symbol, fib_levels=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Candlesticks"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA Fast", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA Slow", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], name="BB High", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"], name="BB Low", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ichimoku_a"], name="Ichimoku A", line=dict(color="purple")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ichimoku_b"], name="Ichimoku B", line=dict(color="gray")))
    if fib_levels:
        for label, level in fib_levels.items():
            fig.add_hline(y=level, line_dash="dash", annotation_text=label, line_color="teal")
    fig.update_layout(title=f"{symbol} Technical Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# ========== ANALISA & TAMPILKAN ========== #
def analyze(symbol):
    df = get_bybit_data(symbol)  # Try Bybit first
    if df.empty:
        st.warning(f"📉 Data Bybit untuk {symbol} tidak ditemukan, mencoba menggunakan Binance...")
        df = get_binance_data(symbol)  # Fallback to Binance
        if df.empty:
            st.error(f"❌ Gagal mendapatkan data untuk {symbol} dari kedua API (Bybit dan Binance).")
            return

    df = add_indicators(df)
    fib_levels = fibonacci_levels(df)
    signal, entry, tp, sl = detect_signal(df)

    st.subheader(f"🔎 Hasil Analisis: {symbol}")
    st.markdown(f"**Sinyal Deteksi:** `{signal}`")
    if signal != "WAIT":
        st.write(f"💰 Entry: `${entry:.2f}`")
        st.write(f"🎯 TP: `${tp:.2f}`")
        st.write(f"🛑 SL: `${sl:.2f}`")

    st.subheader("🧠 Prediksi AI (LSTM Model)")

    # Train model if necessary
    model, scaler = train_lstm_model(df)
    pred_price = predict_future_price(df, model, scaler)
    st.write(f"📈 Prediksi Harga Masa Depan: ${pred_price:.2f}")

    # Plot technical chart
    plot_chart(df, symbol, fib_levels)

# ========== MAIN PROGRAM ========== #
if st.button('Jalankan Analisis'):
    symbols = ["BTCUSDT", "ETHUSDT"]
    for symbol in symbols:
        analyze(symbol)
