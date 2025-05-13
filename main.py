import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import ta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import datetime

st.set_page_config(page_title="AI BTC Signal Analyzer v2", layout="wide")
st.title("🤖 AI BTC Signal Analyzer v2 (Multi-Timeframe + Probabilistic AI)")

@st.cache_data(ttl=60)
def get_kline_data(symbol, interval="3", limit=200):
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
    except:
        return pd.DataFrame()

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    return df.dropna()

def predict_lstm_multi(df, n_steps=30):
    df = df[["close", "rsi", "macd", "ema_fast", "ema_slow"]].dropna()
    if len(df) < n_steps + 1:
        return None, None
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    X = []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i - n_steps:i])
    X = np.array(X)

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    y = data_scaled[n_steps:, 0]
    model.fit(X, y, epochs=15, batch_size=16, verbose=0)

    pred_scaled = model.predict(X[-1].reshape(1, n_steps, X.shape[2]), verbose=0)[0][0]
    predicted_price = scaler.inverse_transform([[pred_scaled, 0, 0, 0, 0]])[0][0]

    direction = "Naik" if predicted_price > df["close"].iloc[-1] else "Turun"
    return direction, predicted_price

def detect_signal_probabilistic(df):
    last = df.iloc[-1]

    score = 0
    total = 4

    # Kondisi teknikal
    if last["ema_fast"] > last["ema_slow"]: score += 1
    if last["macd"] > 0: score += 1
    if 45 <= last["rsi"] <= 65: score += 1
    if last["close"] > df["ema_slow"].iloc[-1]: score += 1

    confidence = (score / total) * 100
    entry = last["close"]
    sl = entry - df["atr"].iloc[-1] * 1.5
    tp = entry + (entry - sl) * 1.5

    if score >= 3:
        signal = "LONG"
    elif score <= 1:
        signal = "SHORT"
        tp = entry - (sl - entry) * 1.5
        sl = entry + df["atr"].iloc[-1] * 1.5
    else:
        signal = "WAIT"
        tp = sl = None

    return signal, entry, tp, sl, confidence


def analyze_pair(symbol):
    df_trend = get_kline_data(symbol, "15")
    df_entry = get_kline_data(symbol, "3")

    if df_trend.empty or df_entry.empty:
        return None

    df_trend = add_indicators(df_trend)
    df_entry = add_indicators(df_entry)

    trend_up = df_trend["ema_fast"].iloc[-1] > df_trend["ema_slow"].iloc[-1] and df_trend["macd"].iloc[-1] > 0
    trend_down = df_trend["ema_fast"].iloc[-1] < df_trend["ema_slow"].iloc[-1] and df_trend["macd"].iloc[-1] < 0

    signal, entry, tp, sl, conf = detect_signal_probabilistic(df_entry)
    lstm_dir, pred_price = predict_lstm_multi(df_entry)

    final_decision = "WAIT"
    reason = "Belum valid"
    if signal == "LONG" and trend_up and lstm_dir == "Naik":
        final_decision = "LONG"
        reason = "✅ Valid naik"
    elif signal == "SHORT" and trend_down and lstm_dir == "Turun":
        final_decision = "SHORT"
        reason = "✅ Valid turun"

    result = {
        "Pair": symbol,
        "Sinyal": final_decision,
        "Confidence": conf,
        "Entry": entry,
        "TP": tp,
        "SL": sl,
        "Valid Sampai": df_entry.index[-1] + pd.Timedelta(minutes=15),
        "Catatan": reason,
        "LSTM Pred": lstm_dir,
        "Chart Data": df_entry
    }
    return result


run_btn = st.button("🚀 RUN ANALISA")

if run_btn:
    st.markdown("## 🔍 Hasil Analisa BTCUSDT & ETHUSDT")
    symbols = ["BTCUSDT", "ETHUSDT"]
    results = []

    for sym in symbols:
        data = analyze_pair(sym)
        if data and data["Sinyal"] in ["LONG", "SHORT"]:
            results.append(data)

    if results:
        df_display = pd.DataFrame(results)[["Pair", "Sinyal", "Entry", "TP", "SL", "Confidence", "Valid Sampai", "Catatan", "LSTM Pred"]]
        st.dataframe(df_display)

        for res in results:
            st.markdown(f"### 📊 Chart {res['Pair']} - Sinyal: {res['Sinyal']} ({res['Confidence']}%)")
            df = res["Chart Data"]

            fig = go.Figure(data=[
                go.Candlestick(
                    x=df.index, open=df["open"], high=df["high"],
                    low=df["low"], close=df["close"], name="Candles"
                ),
                go.Scatter(x=df.index, y=df["ema_fast"], name="EMA Fast", line=dict(color="blue")),
                go.Scatter(x=df.index, y=df["ema_slow"], name="EMA Slow", line=dict(color="orange"))
            ])
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada sinyal valid.")
