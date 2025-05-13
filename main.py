import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import io

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.sidebar.title("🔧 Pengaturan Analisa")

# Sidebar inputs
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0)
interval = st.sidebar.selectbox("⏱️ Pilih Interval Waktu", options=["1", "5", "15", "30", "60", "240", "D"], index=4)
signal_filter = st.sidebar.selectbox("🧳 Filter Sinyal", options=["SEMUA", "LONG", "SHORT", "WAIT"])
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)

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
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def add_indicators(df):
    try:
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
        df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
        df["macd"] = ta.trend.MACD(df["close"]).macd()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return pd.DataFrame()
    return df.dropna()

def detect_signal(df):
    last = df.iloc[-1]
    signal, tp, sl = "WAIT", None, None

    if (
        last["rsi"] < 70 and
        last["ema_fast"] > last["ema_slow"] and
        last["macd"] > 0 and
        last["adx"] > 20 and
        last["stoch_k"] > last["stoch_d"]
    ):
        signal = "LONG"
        tp = last["close"] + 2 * last["atr"]
        sl = last["close"] - 1.5 * last["atr"]
    elif (
        last["rsi"] > 30 and
        last["ema_fast"] < last["ema_slow"] and
        last["macd"] < 0 and
        last["adx"] > 20 and
        last["stoch_k"] < last["stoch_d"]
    ):
        signal = "SHORT"
        tp = last["close"] - 2 * last["atr"]
        sl = last["close"] + 1.5 * last["atr"]

    entry = last["close"]
    return signal, entry, tp, sl

def backtest_signals(df):
    results = []
    for i in range(20, len(df)-10):
        slice_df = df.iloc[i-20:i+1]
        signal, entry, tp, sl = detect_signal(slice_df)
        if signal == "WAIT":
            continue
        future_prices = df.iloc[i+1:i+10]["close"]
        hit_tp = any(p >= tp for p in future_prices) if signal == "LONG" else any(p <= tp for p in future_prices)
        hit_sl = any(p <= sl for p in future_prices) if signal == "LONG" else any(p >= sl for p in future_prices)
        outcome = "TP" if hit_tp else "SL" if hit_sl else "NONE"
        results.append({"index": df.index[i], "signal": signal, "entry": entry, "tp": tp, "sl": sl, "outcome": outcome})
    return pd.DataFrame(results)

@st.cache_resource
def load_lstm_model(path="lstm_model.h5"):
    if os.path.exists(path):
        return load_model(path)
    return None

def predict_lstm(df, n_steps=20):
    df = df[["close"]].dropna()
    if len(df) < n_steps + 1:
        return None, None
    last_close = df["close"].iloc[-1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values.reshape(-1, 1))
    model = load_lstm_model()
    if model is None:
        return None, None
    pred_input = data_scaled[-n_steps:].reshape(1, n_steps, 1)
    pred_scaled = model.predict(pred_input, verbose=0)[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    direction = "Naik" if pred_price > last_close else "Turun"
    return direction, pred_price

def hitung_risk_management(entry, sl, modal, risk_pct):
    risk_dollar = modal * (risk_pct / 100)
    stop_loss_distance = abs(entry - sl)
    if stop_loss_distance == 0:
        return 0, 0, 0, 0
    position_size = risk_dollar / stop_loss_distance
    return round(position_size, 4), round(risk_dollar, 2), round(stop_loss_distance, 2), round(risk_dollar / stop_loss_distance, 2)

def calculate_margin(entry_price, position_size, leverage):
    if leverage == 0:
        return 0
    return round((entry_price * position_size) / leverage, 2)

def show_chart(df, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Candlestick'
    )])
    if 'signal' in df.columns:
        for i in df.index:
            sig = df.loc[i, 'signal']
            price = df.loc[i, 'close']
            if sig == "LONG":
                fig.add_trace(go.Scatter(x=[i], y=[price], mode='markers',
                                         marker=dict(color='green', size=10, symbol='triangle-up'),
                                         name='LONG'))
            elif sig == "SHORT":
                fig.add_trace(go.Scatter(x=[i], y=[price], mode='markers',
                                         marker=dict(color='red', size=10, symbol='triangle-down'),
                                         name='SHORT'))
    fig.update_layout(title=f"{symbol} - Chart & Signals", xaxis_rangeslider_visible=False)
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
        df['signal'] = "WAIT"
        df.loc[df.index[-1], 'signal'] = signal
        
        lstm_dir, lstm_price = predict_lstm(df)
        backtest_df = backtest_signals(df)
        results.append({
            "symbol": symbol,
            "signal": signal,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "lstm": lstm_dir,
            "df": df,
            "backtest": backtest_df
        })
    return results

# Tombol Analisa
if st.sidebar.button("🔍 Jalankan Analisis"):
    st.session_state.analyzed = True
    st.session_state.results = analyze_symbols(["BTCUSDT", "ETHUSDT"], interval=interval)

# Tampilan Hasil
if st.session_state.analyzed:
    st.markdown("<h2 style='color: #4A90E2;'>📊 Hasil Analisa Sinyal</h2>", unsafe_allow_html=True)
    for res in st.session_state.results:
        symbol = res["symbol"]
        if signal_filter != "SEMUA" and res["signal"] != signal_filter:
            continue

        if res["signal"] == "NO DATA":
            st.error(f"{symbol}: Gagal mengambil data.")
            continue

        st.subheader(symbol)
        current_price = res['df']['close'].iloc[-1]
        st.write(f"Harga Sekarang: **${current_price:.2f}**")
        st.write(f"Sinyal: **{res['signal']}**")
        st.write(f"Entry Price: **${res['entry']:.2f}**")
        st.write(f"TP: **${res['tp']:.2f}**" if res['tp'] else "-")
        st.write(f"SL: **${res['sl']:.2f}**" if res['sl'] else "-")
        st.write(f"Prediksi AI (LSTM): **{res['lstm']}**")

        if res['signal'] != "WAIT" and res['tp'] and res['sl']:
            pos_size, loss_amt, sl_dist, rr_ratio = hitung_risk_management(
                res['entry'], res['sl'], modal, risk_pct
            )
            margin = calculate_margin(res['entry'], pos_size, leverage)

            st.write(f"📌 Ukuran Posisi: **{pos_size} USDT**")
            st.write(f"📉 Risiko: ${loss_amt} | SL: ${sl_dist} | R/R Ratio: {rr_ratio:.2f}")
            st.write(f"💼 Estimasi Margin Dibutuhkan: **${margin}** dengan Leverage **{leverage}x**")

        show_chart(res["df"], symbol)

        if isinstance(res.get("backtest"), pd.DataFrame) and not res["backtest"].empty:
            st.write("### ⬅️ Backtest Sinyal")
            outcomes = res["backtest"]["outcome"].value_counts()
            st.write(outcomes)
            acc = (outcomes.get("TP", 0) / outcomes.sum()) * 100
            st.write(f"🎯 Akurasi: **{acc:.2f}%**")
        else:
            st.write("🔁 Backtest tidak tersedia atau data tidak cukup.")

    # Tombol Unduh Excel
    if st.button("📅 Unduh Hasil ke Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for res in st.session_state.results:
                symbol = res["symbol"]
                df = res.get("df")
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(writer, sheet_name=f"{symbol}_data")
                bt_df = res.get("backtest")
                if isinstance(bt_df, pd.DataFrame) and not bt_df.empty:
                    bt_df.to_excel(writer, sheet_name=f"{symbol}_backtest")

        output.seek(0)
        st.download_button(
            label="📥 Klik untuk Mengunduh Excel",
            data=output,
            file_name="hasil_analisa_sinyal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
