import streamlit as st
import pandas as pd
import requests
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI BTC/ETH Signal Analyzer", layout="wide")
st.sidebar.title("🔧 Pengaturan Analisa")

# Sidebar inputs
modal = st.sidebar.number_input("💰 Modal Anda ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("🎯 Risiko per Transaksi (%)", min_value=0.1, max_value=5.0, value=1.0)
interval = st.sidebar.selectbox("⏱️ Pilih Interval Waktu", options=["1", "5", "15", "30", "60", "240", "D"], index=4)
signal_filter = st.sidebar.selectbox("🧳 Filter Sinyal", options=["SEMUA", "LONG", "SHORT", "WAIT"])
leverage = st.sidebar.number_input("⚙️ Leverage", min_value=1, max_value=125, value=10)
mode_strategi = st.sidebar.selectbox("📈 Mode Strategi", ["Adaptif", "Agresif", "Konservatif"])
mode_data = st.sidebar.selectbox("🔄 Mode Data", ["Live", "Backtest"])
symbol = st.sidebar.selectbox("💱 Simbol", options=["BTC", "ETH"], index=0)

uploaded_file = None
df_csv = None
if mode_data == "Backtest":
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Harga BTC/ETH", type=["csv"])
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        df_csv.set_index("timestamp", inplace=True)
        df_csv.sort_index(inplace=True)

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = []

@st.cache_data(ttl=300)
def get_kline_data(symbol, interval="60", limit=200, csv_df=None):
    if csv_df is not None:
        df = csv_df.copy()
    else:
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol + "USDT", "interval": interval, "limit": limit}
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            df = pd.DataFrame(data["result"]["list"])
            df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
            numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df.set_index("timestamp", inplace=True)
        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

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

def plot_chart(df, symbol, signal=None):
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candlesticks"
    ))

    # EMA Lines
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ema_fast'], mode='lines', name="EMA 5", line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ema_slow'], mode='lines', name="EMA 20", line=dict(color='red', width=2)
    ))

    # RSI Indicator
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi'], mode='lines', name="RSI", line=dict(color='purple', width=2)
    ))

    # MACD Indicator
    fig.add_trace(go.Scatter(
        x=df.index, y=df['macd'], mode='lines', name="MACD", line=dict(color='green', width=2)
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_upper'], mode='lines', name="Bollinger Upper", line=dict(color='orange', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_lower'], mode='lines', name="Bollinger Lower", line=dict(color='orange', width=1, dash='dash')
    ))

    # Signals (if available)
    if signal:
        if signal == "LONG":
            fig.add_trace(go.Scatter(
                x=[df.index[-1]], y=[df['close'].iloc[-1]], mode="markers", name="LONG Signal", 
                marker=dict(color="green", size=12, symbol="arrow-bar-up")
            ))
        elif signal == "SHORT":
            fig.add_trace(go.Scatter(
                x=[df.index[-1]], y=[df['close'].iloc[-1]], mode="markers", name="SHORT Signal", 
                marker=dict(color="red", size=12, symbol="arrow-bar-down")
            ))

    fig.update_layout(
        title=f"{symbol} Price Chart with Indicators",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Main code
df = get_kline_data(symbol, interval=interval, csv_df=df_csv)
signal_valid = get_valid_signal(df)

if signal_valid == "LONG":
    entry_price = df['close'].iloc[-1]
    take_profit, stop_loss = calculate_take_profit_and_stop_loss(entry_price, trend="LONG")
    st.write(f"🚀 **Sinyal LONG Valid**: Entry pada ${entry_price:.2f}, Take Profit pada ${take_profit:.2f}, Stop Loss pada ${stop_loss:.2f}")
elif signal_valid == "SHORT":
    entry_price = df['close'].iloc[-1]
    take_profit, stop_loss = calculate_take_profit_and_stop_loss(entry_price, trend="SHORT")
    st.write(f"⚠️ **Sinyal SHORT Valid**: Entry pada ${entry_price:.2f}, Take Profit pada ${take_profit:.2f}, Stop Loss pada ${stop_loss:.2f}")
else:
    st.write("❗ Tidak ada sinyal valid saat ini. Tunggu beberapa saat...")

# Plot the chart with the signal
plot_chart(df, symbol, signal=signal_valid)
