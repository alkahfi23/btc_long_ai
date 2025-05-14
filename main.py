import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
import joblib
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.sidebar.title("🔧 Pengaturan Analisa")

# Sidebar
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", 0.1, 5.0, 1.0)
interval = st.sidebar.selectbox("⏱️ Pilih Interval", ["1", "5", "15", "30", "60", "240", "D"], index=4)
signal_filter = st.sidebar.selectbox("🧳 Filter Sinyal", ["SEMUA", "LONG", "SHORT", "WAIT"])
leverage = st.sidebar.number_input("⚙️ Leverage", 1, 125, 10)
mode_strategi = st.sidebar.selectbox("📈 Mode Strategi", ["Adaptif", "Agresif", "Konservatif"])
mode_data = st.sidebar.selectbox("🔄 Mode Data", ["Live", "Backtest"])

uploaded_file = None
df_csv = None
if mode_data == "Backtest":
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV BTC/ETH", type=["csv"])
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        df_csv.set_index("timestamp", inplace=True)
        df_csv.sort_index(inplace=True)

@st.cache_data(ttl=300)
def get_kline_data(symbol, interval="60", limit=200, csv_df=None):
    if csv_df is not None:
        df = csv_df.copy()
    else:
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            df = pd.DataFrame(data["result"]["list"])
            df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
            df[["open", "high", "low", "close", "volume", "turnover"]] = df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df.set_index("timestamp", inplace=True)
        except Exception as e:
            st.error(f"Gagal fetch {symbol}: {e}")
            return pd.DataFrame()

    df.sort_index(inplace=True)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
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

def load_lstm_model(symbol):
    model_path = f"models/lstm_model_{symbol.lower()}.h5"
    scaler_path = f"models/scaler_{symbol.lower()}.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning(f"⚠️ Model/Scaler tidak ditemukan untuk {symbol}.")
        return None, None
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_lstm(df, model, scaler):
    if len(df) < 60:
        return df["close"].iloc[-1]
    data_scaled = scaler.transform(df[["close"]])
    X = np.array([data_scaled[-60:]])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    pred = model.predict(X, verbose=0)
    return scaler.inverse_transform(pred)[0][0]

def detect_market_regime(df):
    adx = df["adx"]
    ema_fast = df["ema_fast"]
    ema_slow = df["ema_slow"]
    if ema_fast.iloc[-1] > ema_slow.iloc[-1] and adx.iloc[-1] > 20:
        return "UPTREND"
    elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and adx.iloc[-1] > 20:
        return "DOWNTREND"
    return "SIDEWAYS"

def calculate_position_size(entry, stop_loss, modal, risk_pct, leverage):
    risk_amount = modal * (risk_pct / 100)
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit == 0: return 0
    qty = (risk_amount / risk_per_unit) * leverage
    return round(qty, 3)

def detect_signal(df, symbol):
    last = df.iloc[-1]
    entry = last["close"]
    model, scaler = load_lstm_model(symbol)
    lstm_pred = predict_lstm(df, model, scaler) if model and scaler else entry
    pred_up = lstm_pred > entry
    regime = detect_market_regime(df)

    signal, tp, sl = "WAIT", None, None
    if regime in ["UPTREND", "DOWNTREND"]:
        if last["ema_fast"] > last["ema_slow"] and last["rsi"] > 50 and last["macd"] > 0 and last["stoch_k"] > last["stoch_d"]:
            signal, tp, sl = "LONG", entry + 2 * last["atr"], entry - 1.5 * last["atr"]
            if not pred_up: st.warning("⚠️ LSTM bertentangan, prediksi turun")
        elif last["ema_fast"] < last["ema_slow"] and last["rsi"] < 50 and last["macd"] < 0 and last["stoch_k"] < last["stoch_d"]:
            signal, tp, sl = "SHORT", entry - 2 * last["atr"], entry + 1.5 * last["atr"]
            if pred_up: st.warning("⚠️ LSTM bertentangan, prediksi naik")
    elif regime == "SIDEWAYS":
        if last["close"] < last["bb_lower"] and last["stoch_k"] < 20 and last["rsi"] < 30:
            signal, tp, sl = "LONG", entry + 2 * last["atr"], entry - 1.5 * last["atr"]
            if not pred_up: st.warning("⚠️ LSTM bertentangan, prediksi turun")
        elif last["close"] > last["bb_upper"] and last["stoch_k"] > 80 and last["rsi"] > 70:
            signal, tp, sl = "SHORT", entry - 2 * last["atr"], entry + 1.5 * last["atr"]
            if pred_up: st.warning("⚠️ LSTM bertentangan, prediksi naik")

    return signal, entry, tp, sl, pred_up

def plot_chart(df, signal, tp=None, sl=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], line=dict(color='blue'), name='EMA 5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], line=dict(color='orange'), name='EMA 20'))
    if tp: fig.add_hline(y=tp, line=dict(color='green', dash='dash'))
    if sl: fig.add_hline(y=sl, line=dict(color='red', dash='dash'))
    fig.update_layout(title=f"Sinyal: {signal}", xaxis_title="Waktu", yaxis_title="Harga")
    return fig

# === Main Loop ===
for symbol in ["BTCUSDT", "ETHUSDT"]:
    df = get_kline_data(symbol, interval, csv_df=df_csv if mode_data == "Backtest" else None)
    if df.empty: continue
    signal, entry, tp, sl, lstm_up = detect_signal(df, symbol)
    if signal_filter == "SEMUA" or signal == signal_filter:
        st.subheader(f"📊 {symbol}")
        st.plotly_chart(plot_chart(df, signal, tp, sl), use_container_width=True)
        st.metric("Sinyal Strategi", signal)
        st.metric("Prediksi AI (LSTM)", "NAIK" if lstm_up else "TURUN")
        if signal != "WAIT":
            qty = calculate_position_size(entry, sl, modal, risk_pct, leverage)
            trailing_stop = entry - (0.5 * df["atr"].iloc[-1]) if signal == "LONG" else entry + (0.5 * df["atr"].iloc[-1])
            st.write(f"🎯 Entry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f} | Trailing Stop: {trailing_stop:.2f} | Posisi (Qty): {qty}")
