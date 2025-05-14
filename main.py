import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.sidebar.title("🔧 Pengaturan Analisa")

# Sidebar inputs
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0)
interval = st.sidebar.selectbox("⏱️ Pilih Interval Waktu", options=["1", "5", "15", "30", "60", "240", "D"], index=4)
signal_filter = st.sidebar.selectbox("🧳 Filter Sinyal", options=["SEMUA", "LONG", "SHORT", "WAIT"])
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
mode_strategi = st.sidebar.selectbox("📈 Mode Strategi", ["Adaptif", "Agresif", "Konservatif"])

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
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def detect_market_regime(df):
    adx = df["adx"]
    ema_fast = df["ema_fast"]
    ema_slow = df["ema_slow"]
    trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1] and adx.iloc[-1] > 20
    trend_down = ema_fast.iloc[-1] < ema_slow.iloc[-1] and adx.iloc[-1] > 20
    if trend_up:
        return "UPTREND"
    elif trend_down:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

def calculate_position_size(entry_price, stop_loss, modal, risk_pct, leverage):
    risk_amount = modal * (risk_pct / 100)
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        return 0
    qty = (risk_amount / risk_per_unit) * leverage
    return round(qty, 3)

def load_lstm_model():
    try:
        model = load_model("lstm_model.h5")
        return model
    except:
        st.warning("Model LSTM tidak ditemukan. Pastikan model telah dilatih sesuai aset BTCUSDT & ETHUSDT dengan timeframe yang sesuai.")
        return None

def predict_lstm(df, model):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['close']])
    X = np.array([data_scaled[-60:]])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    pred = model.predict(X, verbose=0)
    pred_price = scaler.inverse_transform(pred)[0][0]
    return pred_price

def detect_trend_signal(df, last, entry, lstm_pred):
    pred_up = lstm_pred > entry
    signal = "WAIT"
    tp = sl = None

    if (
        last["ema_fast"] > last["ema_slow"] and
        last["rsi"] > 50 and
        last["macd"] > 0 and
        last["stoch_k"] > last["stoch_d"]
    ):
        signal = "LONG"
        tp = entry + 2 * last["atr"]
        sl = entry - 1.5 * last["atr"]
        if not pred_up:
            st.warning("⚠️ Konflik! Strategi menyarankan LONG, tapi prediksi LSTM menunjukkan penurunan.")
    elif (
        last["ema_fast"] < last["ema_slow"] and
        last["rsi"] < 50 and
        last["macd"] < 0 and
        last["stoch_k"] < last["stoch_d"]
    ):
        signal = "SHORT"
        tp = entry - 2 * last["atr"]
        sl = entry + 1.5 * last["atr"]
        if pred_up:
            st.warning("⚠️ Konflik! Strategi menyarankan SHORT, tapi prediksi LSTM menunjukkan kenaikan.")
    return signal, tp, sl, pred_up

def detect_sideways_signal(df, last, entry, lstm_pred):
    signal = "WAIT"
    tp = sl = None
    pred_up = lstm_pred > entry

    if last["close"] < last["bb_lower"] and last["stoch_k"] < 20 and last["rsi"] < 30:
        signal = "LONG"
        tp = entry + 2 * last["atr"]
        sl = entry - 1.5 * last["atr"]
        if not pred_up:
            st.warning("⚠️ Konflik! Strategi menyarankan LONG, tapi prediksi LSTM menunjukkan penurunan.")
    elif last["close"] > last["bb_upper"] and last["stoch_k"] > 80 and last["rsi"] > 70:
        signal = "SHORT"
        tp = entry - 2 * last["atr"]
        sl = entry + 1.5 * last["atr"]
        if pred_up:
            st.warning("⚠️ Konflik! Strategi menyarankan SHORT, tapi prediksi LSTM menunjukkan kenaikan.")
    return signal, tp, sl, pred_up

def detect_signal(df):
    last = df.iloc[-1]
    regime = detect_market_regime(df)
    entry = last["close"]
    model = load_lstm_model()
    lstm_pred = predict_lstm(df, model) if model else entry

    if regime in ["UPTREND", "DOWNTREND"]:
        return detect_trend_signal(df, last, entry, lstm_pred)
    elif regime == "SIDEWAYS":
        return detect_sideways_signal(df, last, entry, lstm_pred)
    return "WAIT", None, None, lstm_pred > entry

def plot_chart(df, signal, tp=None, sl=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], line=dict(color='blue', width=1), name='EMA 5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], line=dict(color='orange', width=1), name='EMA 20'))
    if tp:
        fig.add_hline(y=tp, line=dict(color="green", dash="dash"), annotation_text="TP")
    if sl:
        fig.add_hline(y=sl, line=dict(color="red", dash="dash"), annotation_text="SL")
    fig.update_layout(title=f"Sinyal: {signal}", xaxis_title='Waktu', yaxis_title='Harga')
    return fig

for symbol in ["BTCUSDT", "ETHUSDT"]:
    df = get_kline_data(symbol, interval=interval)
    if not df.empty:
        signal, entry, tp, sl, lstm_up = detect_signal(df)
        if signal_filter == "SEMUA" or signal == signal_filter:
            st.subheader(f"📊 {symbol}")
            chart = plot_chart(df, signal, tp, sl)
            st.plotly_chart(chart, use_container_width=True)
            st.metric("Sinyal Strategi", signal)
            st.metric("Arah LSTM", "NAIK" if lstm_up else "TURUN")
            if signal != "WAIT":
                qty = calculate_position_size(entry, sl, modal, risk_pct, leverage)
                trailing_stop = entry - (0.5 * df["atr"].iloc[-1]) if signal == "LONG" else entry + (0.5 * df["atr"].iloc[-1])
                st.write(f"Entry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f} | Trailing Stop: {trailing_stop:.2f} | Posisi (Qty): {qty}")
