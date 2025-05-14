import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.sidebar.title("🔧 Pengaturan Analisa")

# Sidebar inputs
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0)
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
symbol = st.sidebar.selectbox("💱 Simbol", options=["BTC", "ETH"], index=0)
timeframe = st.sidebar.selectbox("⏱️ Pilih Timeframe Trading", options=["1", "5", "15", "30", "60", "240", "D"], index=4)

@st.cache_data(ttl=300)
def get_kline_data(symbol, interval="60", limit=200):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol + "USDT", "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return pd.DataFrame()
        df = pd.DataFrame(data["result"]["list"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df.set_index("timestamp", inplace=True)
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

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

def get_valid_signal(df):
    df["signal"] = ""
    df.loc[
        (df["rsi"] < 30) & (df["close"] > df["ema_fast"]) & (df["macd"] > 0),
        "signal"
    ] = "LONG"
    df.loc[
        (df["rsi"] > 70) & (df["close"] < df["ema_fast"]) & (df["macd"] < 0),
        "signal"
    ] = "SHORT"
    valid_signals = df[df["signal"] != ""]
    return valid_signals

def plot_chart(df, entry, stop_loss, take_profit):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Harga Aktual"))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[entry], mode="markers", name="Entry", marker=dict(size=10, color="blue")))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[take_profit], mode="markers", name="Take Profit", marker=dict(size=10, color="green")))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[stop_loss], mode="markers", name="Stop Loss", marker=dict(size=10, color="red")))
    fig.update_layout(title="📈 Chart Harga, Entry, TP, SL", xaxis_title="Waktu", yaxis_title="Harga USD")
    st.plotly_chart(fig, use_container_width=True)

def display_signals(symbol, timeframe):
    with st.spinner(f"🔍 Mengambil data dan menganalisa sinyal untuk {symbol}..."):
        df = get_kline_data(symbol, interval=timeframe)
        if not df.empty:
            valid_signals = get_valid_signal(df)
            if not valid_signals.empty:
                for idx, row in valid_signals.tail(3).iterrows():
                    entry = row["close"]
                    sl_pct = 0.02
                    tp_pct = 0.03

                    stop_loss = entry * (1 - sl_pct) if row["signal"] == "LONG" else entry * (1 + sl_pct)
                    take_profit = entry * (1 + tp_pct) if row["signal"] == "LONG" else entry * (1 - tp_pct)

                    stop_range = abs(entry - stop_loss)
                    risk_amount = modal * (risk_pct / 100)
                    position_size = (risk_amount / stop_range) * leverage

                    potential_profit = abs(take_profit - entry) * position_size
                    potential_loss = abs(entry - stop_loss) * position_size

                    st.markdown(f"""
                        ### 📍 {symbol} - Sinyal **{row['signal']}**
                        ⏱️ Waktu: `{idx.strftime('%Y-%m-%d %H:%M')}`  
                        💵 Entry: **${entry:.2f}**  
                        🛑 Stop Loss: **${stop_loss:.2f}**  
                        🎯 Take Profit: **${take_profit:.2f}**  
                        📏 Ukuran Posisi: **{position_size:.4f} USDT**  
                        ✅ Potensi Profit: **${potential_profit:.2f}**  
                        ❌ Potensi Rugi: **${potential_loss:.2f}**
                    """)
                    plot_chart(df, entry, stop_loss, take_profit)
            else:
                st.info(f"Tidak ada sinyal valid untuk {symbol}")
        else:
            st.warning(f"Gagal mengambil data untuk {symbol}")

# Simulasi Pertumbuhan Portofolio
def simulate_portfolio_growth(df, signals, initial_balance, leverage, risk_pct):
    balance = initial_balance
    equity_curve = []
    timestamps = []

    for idx, row in signals.iterrows():
        entry_price = row["close"]
        direction = row["signal"]
        stop_pct = 0.02
        tp_pct = 0.03

        sl_price = entry_price * (1 - stop_pct) if direction == "LONG" else entry_price * (1 + stop_pct)
        tp_price = entry_price * (1 + tp_pct) if direction == "LONG" else entry_price * (1 - tp_pct)
        
        future_data = df[df.index > idx]
        result = "HOLD"

        for future_idx, future_row in future_data.iterrows():
            high = future_row["high"]
            low = future_row["low"]
            if direction == "LONG":
                if low <= sl_price:
                    result = "SL"
                    exit_price = sl_price
                    break
                elif high >= tp_price:
                    result = "TP"
                    exit_price = tp_price
                    break
            else:  # SHORT
                if high >= sl_price:
                    result = "SL"
                    exit_price = sl_price
                    break
                elif low <= tp_price:
                    result = "TP"
                    exit_price = tp_price
                    break

        risk_amount = balance * (risk_pct / 100)
        stop_range = abs(entry_price - sl_price)
        position_size = (risk_amount / stop_range) * leverage

        if result == "TP":
            pnl = (abs(tp_price - entry_price)) * position_size
        elif result == "SL":
            pnl = -(abs(entry_price - sl_price)) * position_size
        else:
            pnl = 0

        balance += pnl
        equity_curve.append(balance)
        timestamps.append(future_idx if result != "HOLD" else df.index[-1])

    return pd.DataFrame({"Waktu": timestamps, "Saldo": equity_curve})

# Tombol analisa sinyal
if st.sidebar.button("🚀 Mulai Analisis Sinyal", use_container_width=True):
    display_signals("BTC", timeframe)
    display_signals("ETH", timeframe)

# Tombol simulasi portofolio
if st.sidebar.button("📊 Simulasikan Pertumbuhan Portofolio", use_container_width=True):
    df = get_kline_data(symbol, interval=timeframe)
    if not df.empty:
        signals = get_valid_signal(df)
        if not signals.empty:
            st.subheader(f"📈 Simulasi Pertumbuhan Portofolio - {symbol}")
            sim_df = simulate_portfolio_growth(df, signals, modal, leverage, risk_pct)
            if not sim_df.empty:
                st.line_chart(sim_df.set_index("Waktu"), use_container_width=True)
                st.success(f"Saldo akhir dari simulasi: ${sim_df['Saldo'].iloc[-1]:.2f}")
            else:
                st.warning("Simulasi tidak menghasilkan perubahan portofolio.")
        else:
            st.warning("Tidak ada sinyal untuk disimulasikan.")
    else:
        st.error("Gagal mengambil data untuk simulasi.")
