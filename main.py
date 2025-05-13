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
import datetime

# ========== KONFIGURASI ==========
st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("🤖 AI BTC Signal Analyzer (Multi-Timeframe Strategy)")
st.markdown("Klik tombol di bawah untuk menjalankan analisa.")
run_button = st.button("🚀 RUN ANALISA")

# ========== API ==========
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
    except Exception as e:
        st.error(f"Gagal mengambil data {symbol}: {e}")
        return pd.DataFrame()

# ========== INDICATORS ==========
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

# ========== LSTM ==========
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
    predicted_scaled = model.predict(input_data, verbose=0)[0][0]
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
    if df_trend.empty or df_entry.empty:
        return "NO DATA", None, None, None, df_entry, "❌ Data tidak tersedia"

    df_trend = add_indicators(df_trend)
    df_entry = add_indicators(df_entry)
    trend = df_trend.iloc[-1]
    trend_long = trend["ema_fast"] > trend["ema_slow"] and trend["macd"] > 0
    trend_short = trend["ema_fast"] < trend["ema_slow"] and trend["macd"] < 0

    signal, entry, tp, sl = detect_signal(df_entry)
    lstm_dir, _ = predict_lstm(df_entry)

    if signal == "LONG" and trend_long and lstm_dir == "Naik":
        pass
    elif signal == "SHORT" and trend_short and lstm_dir == "Turun":
        pass
    else:
        return "WAIT", None, None, None, df_entry, "Belum terkonfirmasi"

    # Risk check
    if entry and sl and tp:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        risk_reward = reward / risk if risk != 0 else 0
        sl_pct = abs(entry - sl) / entry * 100

        if risk_reward < 1 or sl_pct < 0.5:
            return "WAIT", None, None, None, df_entry, "⚠️ Risiko tinggi (SL terlalu dekat atau R:R < 1)"

    return signal, entry, tp, sl, df_entry, "✅ Valid"

# ========== ANALISIS DAN VISUALISASI ==========
if run_button:
    symbols = ["BTCUSDT", "ETHUSDT"]
    st.markdown("## 📊 Sinyal Valid (BTCUSDT & ETHUSDT)")
    summary = []

    for sym in symbols:
        sig, ent, tp, sl, df_sym, note = analyze_multi_timeframe(sym, tf_trend="15", tf_entry="3")
        if sig in ["LONG", "SHORT"] and not df_sym.empty:
            last = df_sym.iloc[-1]
            signal_time = last.name
            valid_until = signal_time + pd.Timedelta(minutes=15)

            strength = abs(last["ema_fast"] - last["ema_slow"]) + abs(last["macd"]) + abs(last["rsi"] - 50)
            summary.append({
                "Pair": sym,
                "Sinyal": sig,
                "Entry": f"${ent:.2f}" if ent else "-",
                "TP": f"${tp:.2f}" if tp else "-",
                "SL": f"${sl:.2f}" if sl else "-",
                "RSI": round(last["rsi"], 2),
                "MACD": round(last["macd"], 4),
                "EMA Fast": round(last["ema_fast"], 2),
                "EMA Slow": round(last["ema_slow"], 2),
                "Sinyal Valid Sampai": valid_until.strftime("%Y-%m-%d %H:%M:%S"),
                "Catatan Risiko": note,
                "Kekuatan Sinyal": strength,
                "Chart Data": df_sym
            })

    if summary:
        df_summary = pd.DataFrame(summary)
        df_display = df_summary.drop(columns=["Kekuatan Sinyal", "Chart Data"])
        df_display = df_display.sort_values(by="Kekuatan Sinyal", ascending=False)
        st.dataframe(df_display)

        # ========== CHART ==========
        for row in summary:
            st.markdown(f"### 📈 {row['Pair']} ({row['Sinyal']})")
            df_candle = row["Chart Data"].copy()
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df_candle.index,
                    open=df_candle["open"],
                    high=df_candle["high"],
                    low=df_candle["low"],
                    close=df_candle["close"],
                    name="Candles"
                ),
                go.Scatter(x=df_candle.index, y=df_candle["ema_fast"], line=dict(color='blue', width=1), name="EMA Fast"),
                go.Scatter(x=df_candle.index, y=df_candle["ema_slow"], line=dict(color='orange', width=1), name="EMA Slow"),
                go.Scatter(x=df_candle.index, y=df_candle["bb_high"], line=dict(color='green', width=1, dash='dot'), name="BB High"),
                go.Scatter(x=df_candle.index, y=df_candle["bb_low"], line=dict(color='red', width=1, dash='dot'), name="BB Low"),
            ])
            fig.update_layout(
                title=f"{row['Pair']} Candlestick + Indikator (Valid Sampai: {row['Sinyal Valid Sampai']})",
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Belum ada sinyal valid untuk BTCUSDT dan ETHUSDT saat ini.")

