import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import ta
from datetime import datetime

# ========== KONFIGURASI ========== #
st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("🤖 AI BTC Signal Analyzer (Multi-Timeframe Strategy)")

# ========== API ========== #
@st.cache_data(ttl=60)
def get_historical_data(symbol, interval="1", limit=500):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data["result"]["list"], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# ========== INDICATORS ========== #
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    return df

# ========== LSTM ========== #
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout untuk mencegah overfitting
    model.add(Dense(1))  # Layer output untuk prediksi harga
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def preprocess_data(df, n_steps=30):
    df = df[["close", "rsi", "macd", "ema_fast", "ema_slow"]].dropna()  # Pilih fitur yang akan digunakan
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)  # Normalisasi data

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])  # Ambil data sebelumnya untuk input
        y.append(scaled_data[i, 0])  # Target adalah harga close

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Fungsi untuk memuat model dan melakukan prediksi
def predict_lstm_trained(df, model, scaler, n_steps=30):
    df = df[["close", "rsi", "macd", "ema_fast", "ema_slow"]].dropna()
    data_scaled = scaler.transform(df.values)
    
    # Membuat input sesuai dengan panjang langkah yang diperlukan
    X = []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i-n_steps:i])
    
    X = np.array(X)

    # Prediksi harga
    predicted_scaled = model.predict(X[-1].reshape(1, n_steps, X.shape[2]))[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled, 0, 0, 0, 0]])[0][0]
    
    return predicted_price

# ========== PREDIKSI MODEL LSTM ========== #
btc_data = get_historical_data("BTCUSDT", interval="1", limit=500)
eth_data = get_historical_data("ETHUSDT", interval="1", limit=500)

btc_data = add_indicators(btc_data)
eth_data = add_indicators(eth_data)

# Preprocessing untuk LSTM
X_btc, y_btc, btc_scaler = preprocess_data(btc_data)
X_eth, y_eth, eth_scaler = preprocess_data(eth_data)

# Membuat dan melatih model LSTM untuk BTC dan ETH
btc_model = create_lstm_model((X_btc.shape[1], X_btc.shape[2]))
eth_model = create_lstm_model((X_eth.shape[1], X_eth.shape[2]))

# Latih model
btc_model.fit(X_btc, y_btc, epochs=30, batch_size=16, verbose=1)
eth_model.fit(X_eth, y_eth, epochs=30, batch_size=16, verbose=1)

# Simpan model yang telah dilatih
btc_model.save("btc_model.h5")
eth_model.save("eth_model.h5")

st.write("Model telah dilatih dan disimpan.")

# Memuat model yang telah dilatih
btc_model = load_model("btc_model.h5")
eth_model = load_model("eth_model.h5")

# Prediksi untuk BTC dan ETH
btc_pred = predict_lstm_trained(btc_data, btc_model, btc_scaler)
eth_pred = predict_lstm_trained(eth_data, eth_model, eth_scaler)

# Menampilkan hasil prediksi
st.write(f"Prediksi harga BTC: {btc_pred:.2f}")
st.write(f"Prediksi harga ETH: {eth_pred:.2f}")

# ========== TAMPILKAN CHART CANDLESTICK ========== #
def plot_candle_chart(df, symbol):
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candles"),
                         go.Scatter(x=df.index, y=df["ema_fast"], name="EMA Fast", line=dict(color="blue")),
                         go.Scatter(x=df.index, y=df["ema_slow"], name="EMA Slow", line=dict(color="orange"))])
    fig.update_layout(title=f"Chart {symbol}", xaxis_rangeslider_visible=False)
    fig.show()

plot_candle_chart(btc_data, "BTCUSDT")
plot_candle_chart(eth_data, "ETHUSDT")

# ========== DETEKSI SINYAL ========== #
def detect_signal(df):
    if df.empty: return "WAIT", None, None, None
    last = df.iloc[-1]
    long_cond = (
        last["rsi"] < 70 and
        last["ema_fast"] > last["ema_slow"] and
        last["macd"] > 0 and
        last["close"] > last["bb_low"]
    )
    short_cond = (
        last["rsi"] > 30 and
        last["ema_fast"] < last["ema_slow"] and
        last["macd"] < 0 and
        last["close"] < last["bb_high"]
    )
    entry = last["close"]
    if long_cond:
        return "LONG", entry, entry * 1.02, entry * 0.985
    elif short_cond:
        return "SHORT", entry, entry * 0.98, entry * 1.015
    else:
        return "WAIT", None, None, None

# ========== ANALISIS SIGNAL ========== #
def analyze_signal(symbol):
    df = get_historical_data(symbol, interval="1", limit=500)
    df = add_indicators(df)
    signal, entry, tp, sl = detect_signal(df)
    return signal, entry, tp, sl, df

# Tombol untuk menjalankan analisis
if st.button("Run Analysis"):
    signals = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        signal, entry, tp, sl, df = analyze_signal(symbol)
        signals.append({
            "Pair": symbol,
            "Signal": signal,
            "Entry": entry,
            "TP": tp,
            "SL": sl
        })

    # Tampilkan hasil sinyal dalam tabel
    if signals:
        df_summary = pd.DataFrame(signals)
        st.dataframe(df_summary)
    else:
        st.write("No valid signals found.")
