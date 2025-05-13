import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta
import numpy as np

st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")

# Ambil semua simbol
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

# Judul
st.title("📊 AI BTC Signal Analyzer (LONG & SHORT)")

# Sidebar simbol
symbols = get_all_symbols()
symbol = st.sidebar.selectbox("🔄 Pilih Pair Trading:", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)

# Ambil data candle untuk dua timeframe: 1m dan 15m
@st.cache_data(ttl=60)
def get_kline_data(symbol, interval="1", limit=100):
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
            st.error("❌ Data API Bybit tidak valid.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal ambil data: {e}")
        return pd.DataFrame()

df_1m = get_kline_data(symbol, interval="1")
df_15m = get_kline_data(symbol, interval="15")

# Tambah indikator

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["bop"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    return df

if not df_1m.empty:
    df_1m = add_indicators(df_1m)
if not df_15m.empty:
    df_15m = add_indicators(df_15m)

# Deteksi sinyal berbasis 2 timeframe

def detect_signal(df_entry, df_confirm):
    if df_entry.empty or df_confirm.empty:
        return "NO DATA", None, None, None

    last_entry = df_entry.iloc[-1]
    last_confirm = df_confirm.iloc[-1]

    long_condition = (
        last_entry["rsi"] < 70 and
        last_entry["ema_fast"] > last_entry["ema_slow"] and
        last_entry["macd"] > 0 and
        last_entry["bop"] > 0 and
        last_confirm["ema_fast"] > last_confirm["ema_slow"]
    )

    short_condition = (
        last_entry["rsi"] > 30 and
        last_entry["ema_fast"] < last_entry["ema_slow"] and
        last_entry["macd"] < 0 and
        last_entry["bop"] < 0 and
        last_confirm["ema_fast"] < last_confirm["ema_slow"]
    )

    if long_condition:
        entry = last_entry["close"]
        sl = entry * 0.99
        tp = entry * 1.02
        return "LONG", entry, tp, sl
    elif short_condition:
        entry = last_entry["close"]
        sl = entry * 1.01
        tp = entry * 0.98
        return "SHORT", entry, tp, sl
    else:
        return "WAIT", None, None, None

signal, entry_price, take_profit, stop_loss = detect_signal(df_1m, df_15m)

# Hitung posisi

def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0):
    risk_amount = balance * (risk_pct / 100)
    stop_loss_range = abs(entry - sl)
    qty = risk_amount / (stop_loss_range / entry)
    max_qty = (balance * leverage) / entry
    return round(min(qty, max_qty), 3)

# Input user
balance = st.sidebar.number_input("💰 Modal (USDT):", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("⚙️ Leverage", 1, 100, 10)

# Chart

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
    fig.update_layout(title=f"📉 Grafik {symbol}", xaxis_rangeslider_visible=False, height=500)
    return fig

if not df_1m.empty:
    st.plotly_chart(plot_chart(df_1m), use_container_width=True)
    st.markdown(f"### 🤖 Sinyal AI: **{signal}**")

    if signal in ["LONG", "SHORT"]:
        position_size = calculate_position_size(balance, entry_price, stop_loss, leverage)
        arah = "📈 LONG (Naik)" if signal == "LONG" else "📉 SHORT (Turun)"
        st.markdown(f"""
**🧭 Arah:** {arah}  
🎯 **Entry Price:** `${entry_price:.2f}`  
🛑 *StopLoss:* `${stop_loss:.2f}`  
✅ **Take Profit:** `${take_profit:.2f}`  
📦 **Position Size (saran):** `{position_size}` kontrak {symbol}  
*(leverage {leverage}x, risiko 1% dari modal)*
        """)
    else:
        st.info("⏳ Belum ada sinyal. AI menunggu setup ideal.")
else:
    st.error("❌ Data tidak tersedia.")
