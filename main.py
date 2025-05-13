
# ========== IMPORT ==========
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import ta
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ========== KONFIGURASI ==========
st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("🤖 AI BTC Signal Analyzer (Multi-Timeframe Strategy)")

# ========== FUNGSI API ==========
@st.cache_data(ttl=3600)
def get_all_symbols():
    url = "https://api.bybit.com/v5/market/instruments-info"
    try:
        res = requests.get(url, params={"category": "linear"}, timeout=10)
        data = res.json()
        return sorted([i["symbol"] for i in data["result"]["list"] if "USDT" in i["symbol"]])
    except:
        return ["BTCUSDT"]

@st.cache_data(ttl=60)
def get_kline_data(symbol, interval="1", limit=100):
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

# ========== INDICATOR ==========
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["close"]).stochrsi()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    return df

# ========== PREDIKSI LSTM ==========
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
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    input_data = data_scaled[-n_steps:].reshape((1, n_steps, 1))
    predicted_scaled = model.predict(input_data)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

    direction = "Naik" if predicted_price > df["close"].iloc[-1] else "Turun"
    return direction, predicted_price

# ========== SINYAL ==========
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

def analyze_multi_timeframe(symbol, tf_trend="15", tf_entry="3"):
    df_trend = get_kline_data(symbol, tf_trend)
    df_entry = get_kline_data(symbol, tf_entry)
    if df_trend.empty or df_entry.empty: return "NO DATA", None, None, None, df_entry

    df_trend = add_indicators(df_trend)
    df_entry = add_indicators(df_entry)
    trend = df_trend.iloc[-1]
    trend_long = trend["ema_fast"] > trend["ema_slow"] and trend["macd"] > 0
    trend_short = trend["ema_fast"] < trend["ema_slow"] and trend["macd"] < 0

    signal, entry, tp, sl = detect_signal(df_entry)
    lstm_dir, _ = predict_lstm(df_entry)

    if signal == "LONG" and trend_long and lstm_dir == "Naik":
        return signal, entry, tp, sl, df_entry
    elif signal == "SHORT" and trend_short and lstm_dir == "Turun":
        return signal, entry, tp, sl, df_entry
    else:
        return "WAIT", None, None, None, df_entry



# ========== DAFTAR SINYAL SEMUA PAIR ==========
st.markdown("## 📊 Daftar Sinyal Valid (LONG / SHORT)")
summary = []

for sym in symbols[:20]:  # Batasi 20 pair teratas agar tidak overload
    sig, ent, tp, sl, df_sym = analyze_multi_timeframe(sym, tf_trend="15", tf_entry=entry_tf)
    lstm_dir, _ = predict_lstm(df_sym)
    if sig in ["LONG", "SHORT"] and lstm_dir and ((sig == "LONG" and lstm_dir == "Naik") or (sig == "SHORT" and lstm_dir == "Turun")):
        last = df_sym.iloc[-1]
        strength = abs(last["ema_fast"] - last["ema_slow"]) + abs(last["macd"]) + abs(last["rsi"] - 50)
        summary.append({
            "Pair": sym,
            "Sinyal": sig,
            "Entry": f"${ent:.2f}",
            "TP": f"${tp:.2f}",
            "SL": f"${sl:.2f}",
            "LSTM": lstm_dir,
            "Kekuatan Sinyal": strength
        })

if summary:
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values(by="Kekuatan Sinyal", ascending=False)
    st.dataframe(df_summary.drop(columns=["Kekuatan Sinyal"]))
else:
    st.info("Belum ada sinyal valid dari semua pair saat ini.")

