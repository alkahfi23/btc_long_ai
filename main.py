import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.title("🤖 AI BTC/ETH Signal Analyzer (Full Version)")

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
        df = pd.DataFrame(data["result"]["list"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    return df

def detect_signal(df):
    last = df.iloc[-1]
    signal = "WAIT"
    tp, sl = None, None
    entry = last["close"]
    if last["rsi"] < 70 and last["ema_fast"] > last["ema_slow"] and last["macd"] > 0:
        signal = "LONG"
        tp = entry * 1.02
        sl = entry * 0.985
    elif last["rsi"] > 30 and last["ema_fast"] < last["ema_slow"] and last["macd"] < 0:
        signal = "SHORT"
        tp = entry * 0.98
        sl = entry * 1.015
    return signal, entry, tp, sl

def predict_lstm(df, n_steps=20):
    df = df[["close"]].dropna()
    if len(df) < n_steps + 1:
        return None, None
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i - n_steps:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    pred_input = data_scaled[-n_steps:].reshape(1, n_steps, 1)
    pred_scaled = model.predict(pred_input)[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    direction = "Naik" if pred_price > df["close"].iloc[-1] else "Turun"
    return direction, pred_price

def show_chart(df, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Candles'
    )])
    fig.update_layout(title=f'Chart {symbol}', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def analyze_symbols(symbols, interval="60"):
    results = []
    for symbol in symbols:
        df = get_kline_data(symbol, interval=interval)
        if df.empty:
            results.append({"symbol": symbol, "signal": "NO DATA", "df": None})
            continue
        df = add_indicators(df)
        signal, entry, tp, sl = detect_signal(df)
        lstm_dir, lstm_price = predict_lstm(df)
        results.append({
            "symbol": symbol, "signal": signal, "entry": entry,
            "tp": tp, "sl": sl, "lstm": lstm_dir, "df": df
        })
    return results

if st.button("🔍 Jalankan Analisis"):
    st.session_state.analyzed = True
    st.session_state.results = analyze_symbols(["BTCUSDT", "ETHUSDT"], interval="60")

if st.session_state.analyzed:
    st.subheader("📊 Hasil Analisa Sinyal Lengkap")
    for res in st.session_state.results:
        sym = res["symbol"]
        if res["signal"] == "NO DATA":
            st.error(f"{sym}: Gagal mengambil data.")
            continue
        st.markdown(f"### {sym}")
        st.write(f"Sinyal: **{res['signal']}**")
        st.write(f"Entry: ${res['entry']:.2f}")
        st.write(f"TP: ${res['tp']:.2f}" if res['tp'] else "-")
        st.write(f"SL: ${res['sl']:.2f}" if res['sl'] else "-")
        st.write(f"Prediksi AI LSTM: **{res['lstm']}**")
        show_chart(res["df"], sym)
