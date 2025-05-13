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
from keras.callbacks import EarlyStopping

# ========== KONFIGURASI STREAMLIT ==========
st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("🤖 AI BTC Signal Analyzer (Multi-Timeframe Strategy)")

# ========== API DATA ==========
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
        if "result" not in data or "list" not in data["result"]:
            return pd.DataFrame()
        df = pd.DataFrame(data["result"]["list"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except:
        return pd.DataFrame()

# ========== INDICATORS ==========
def add_indicators(df):
    if df.empty:
        return df
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["close"]).stochrsi()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    return df

# ========== LSTM PREDIKSI ==========
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
    es = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=10, batch_size=16, verbose=0, callbacks=[es])

    input_data = data_scaled[-n_steps:].reshape((1, n_steps, 1))
    predicted_scaled = model.predict(input_data, verbose=0)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    direction = "Naik" if predicted_price > df["close"].iloc[-1] else "Turun"
    return direction, predicted_price

# ========== SIGNAL DETECTION ==========
def detect_signal(df):
    if df.empty:
        return "WAIT", None, None, None
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
        return "NO DATA", None, None, None, df_entry

    df_trend = add_indicators(df_trend)
    df_entry = add_indicators(df_entry)
    trend = df_trend.iloc[-1]
    trend_long = trend["ema_fast"] > trend["ema_slow"] and trend["macd"] > 0
    trend_short = trend["ema_fast"] < trend["ema_slow"] and trend["macd"] < 0

    signal, entry, tp, sl = detect_signal(df_entry)
    if signal == "LONG" and trend_long:
        return signal, entry, tp, sl, df_entry
    elif signal == "SHORT" and trend_short:
        return signal, entry, tp, sl, df_entry
    else:
        return "WAIT", None, None, None, df_entry

# ========== POSITION SIZE ==========
def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0, min_sl_pct=0.7):
    if entry == 0 or sl == 0:
        return 0.0
    stop_range = abs(entry - sl)
    if (stop_range / entry) * 100 < min_sl_pct:
        st.warning(f"⚠️ Stop Loss terlalu dekat ({(stop_range / entry) * 100:.2f}%). Risiko tinggi.")
        return 0.0
    risk_amount = balance * (risk_pct / 100)
    qty = risk_amount / (stop_range / entry)
    max_qty = (balance * leverage) / entry
    return round(min(qty, max_qty) * 0.9, 3)

# ========== TRAILING STOP ==========
def calculate_trailing_stop(entry, volatility, direction="LONG", multiplier=1.5):
    return entry - volatility * multiplier if direction == "LONG" else entry + volatility * multiplier

# ========== VOLATILITY ==========
def estimate_volatility(df, window=14):
    if df.empty or len(df) < window:
        return 0.0
    returns = df["close"].pct_change()
    return (returns.rolling(window=window).std() * 100).iloc[-1]

# ========== PLOT CHART ==========
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA 5", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA 21", line=dict(color="orange")))
    fig.update_layout(title="📉 Grafik Candlestick", height=500)
    return fig

# ========== UI STREAMLIT ==========
symbols = get_all_symbols()
symbol = st.sidebar.selectbox("🔄 Pilih Pair", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
entry_tf = st.sidebar.selectbox("⏱️ Timeframe Entry", ["1", "3", "5", "15", "30", "60"], index=1)
balance = st.sidebar.number_input("💰 Modal (USDT)", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("⚙️ Leverage", 1, 100, 10)

# ========== EKSEKUSI ==========
signal, entry_price, take_profit, stop_loss, df_plot = analyze_multi_timeframe(symbol, tf_trend="15", tf_entry=entry_tf)
lstm_dir, lstm_pred = predict_lstm(df_plot)

st.subheader(f"📡 Sinyal AI (Multi TF + LSTM): **{signal}** | Prediksi: **{lstm_dir}** (${lstm_pred:.4f})" if lstm_pred else f"📡 Sinyal AI: {signal}")

if signal in ["LONG", "SHORT"]:
    volatility = estimate_volatility(df_plot)
    trailing_sl = calculate_trailing_stop(entry_price, volatility, signal)
    position = calculate_position_size(balance, entry_price, trailing_sl, leverage)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Entry", f"${entry_price:.4f}")
        st.metric("✅ Take Profit", f"${take_profit:.4f}")
    with col2:
        st.metric("🛑 Trailing Stop", f"${trailing_sl:.4f}")
        st.metric("📦 Posisi", f"{position} kontrak")
    st.caption(f"(Leverage {leverage}x | Modal ${balance:.2f})")
else:
    st.info("⏳ AI menunggu sinyal ideal...")

if not df_plot.empty:
    st.plotly_chart(plot_chart(df_plot), use_container_width=True)
    st.markdown("### 📌 Ringkasan Indikator")
    st.dataframe(df_plot[["close", "rsi", "ema_fast", "ema_slow", "macd"]].tail(5).round(2))
