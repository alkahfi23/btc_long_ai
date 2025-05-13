import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta
import numpy as np

st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")

@st.cache_data(ttl=3600)
def get_all_symbols():
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("retCode") == 0 and "result" in data and "list" in data["result"]:
            return sorted([item["symbol"] for item in data["result"]["list"] if "USDT" in item["symbol"]])
        else:
            return ["BTCUSDT"]
    except Exception as e:
        st.warning(f"⚠️ Gagal mengambil simbol: {e}")
        return ["BTCUSDT"]

st.title("📊 AI BTC Signal Analyzer (LONG & SHORT)")

symbols = get_all_symbols()
symbol = st.sidebar.selectbox("🔄 Pilih Pair Trading:", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
interval = "1"
limit = 100

@st.cache_data(ttl=60)
def get_kline_data(symbol, interval, limit):
    url = f"https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "result" in data and "list" in data["result"]:
            df = pd.DataFrame(data["result"]["list"], columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df.astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        else:
            st.error("❌ Data API Bybit tidak valid atau diblokir. Coba pakai VPN atau periksa parameter.")
            st.write("Debug response:", data)
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal mengambil data dari API Bybit: {e}")
        return pd.DataFrame()

df = get_kline_data(symbol, interval, limit)

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    return df

if not df.empty:
    df = add_indicators(df)

# === AI Analysis Start ===
entry_price = float(df['close'].iloc[-1])
signal = "LONG" if close[-1] > close[-2] else "SHORT"

if signal == "LONG":
    take_profit = entry_price * 1.01
    stop_loss = entry_price * 0.99
else:
    take_profit = entry_price * 0.99
    stop_loss = entry_price * 1.01

# Format harga
entry_price_fmt = f"${entry_price:,.2f}"
take_profit_fmt = f"${take_profit:,.2f}"
stop_loss_fmt = f"${stop_loss:,.2f}"
# === AI Analysis End ===

def detect_signal(df):
    if df.empty:
        return "NO DATA", None, None, None

    last = df.iloc[-1]
    long_condition = (
        last["rsi"] < 70 and
        last["ema_fast"] > last["ema_slow"] and
        last["macd"] > 0
    )
    short_condition = (
        last["rsi"] > 30 and
        last["ema_fast"] < last["ema_slow"] and
        last["macd"] < 0
    )

    if long_condition:
        entry = last["close"]
        stop_loss = entry - (entry * 0.01)
        take_profit = entry + (entry * 0.02)
        return "LONG", entry, take_profit, stop_loss
    elif short_condition:
        entry = last["close"]
        stop_loss = entry + (entry * 0.01)
        take_profit = entry - (entry * 0.02)
        return "SHORT", entry, take_profit, stop_loss
    else:
        return "WAIT", None, None, None

signal, entry_price, tp, sl = detect_signal(df)

def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0):
    risk_amount = balance * (risk_pct / 100)
    stop_loss_range = abs(entry - sl)
    qty = risk_amount / (stop_loss_range / entry)
    max_qty = (balance * leverage) / entry
    return round(min(qty, max_qty), 3)

balance = st.sidebar.number_input("💰 Masukkan modal (USDT):", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("⚙️ Leverage", 1, 100, 10)

def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], line=dict(color="blue"), name="EMA Fast"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], line=dict(color="orange"), name="EMA Slow"))
    fig.update_layout(title="📊 BTC/USDT Chart", xaxis_rangeslider_visible=False, height=500)
    return fig

if not df.empty:
    st.plotly_chart(plot_chart(df), use_container_width=True)
    st.markdown(f"### 📈 Sinyal AI: **{signal}**")
    if signal in ["LONG", "SHORT"]:
        position_size = calculate_position_size(balance, entry_price, sl, leverage)
        arah = "📈 LONG (Naik)" if signal == "LONG" else "📉 SHORT (Turun)"
        st.markdown(f'''
        **🧭 Arah:** {arah}  
        st.markdown(f"🎯 **Entry Price:** {entry_price_fmt} 
        🛑 *StopLoss* : {stop_loss_fmt}")
        st.markdown(f"✅ **Take Profit:** {take_profit_fmt}")  
        **📦 Position Size (saran):** {position_size} kontrak BTC  
        (dengan leverage {leverage}x dan risiko 1% dari modal)
        ''')
    else:
        st.warning("📡 Belum ada sinyal masuk posisi. AI masih menunggu kondisi ideal.")
else:
    st.error("⚠️ Data tidak tersedia atau gagal diambil.")
