import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta

# ================== CONFIG & SETUP ==================
st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("📊 AI BTC Signal Analyzer (LONG & SHORT)")

# ================== FUNGSI API ==================
@st.cache_data(ttl=3600)
def get_all_symbols():
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return sorted([item["symbol"] for item in data["result"]["list"] if "USDT" in item["symbol"]])
    except:
        return ["BTCUSDT"]

@st.cache_data(ttl=60)
def get_kline_data(symbol, interval="1", limit=100):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data["result"]["list"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except:
        return pd.DataFrame()

# ================== FUNGSI TEKNIKAL ==================
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    return df

def detect_signal(df):
    if df.empty:
        return "NO DATA", None, None, None
    last = df.iloc[-1]
    long_cond = (last["rsi"] < 70 and last["ema_fast"] > last["ema_slow"] and last["macd"] > 0)
    short_cond = (last["rsi"] > 30 and last["ema_fast"] < last["ema_slow"] and last["macd"] < 0)
    if long_cond:
        entry, sl, tp = last["close"], last["close"] * 0.99, last["close"] * 1.02
        return "LONG", entry, tp, sl
    elif short_cond:
        entry, sl, tp = last["close"], last["close"] * 1.01, last["close"] * 0.98
        return "SHORT", entry, tp, sl
    else:
        return "WAIT", None, None, None

def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0):
    risk_amount = balance * (risk_pct / 100)
    stop_loss_range = abs(entry - sl)
    qty = risk_amount / (stop_loss_range / entry)
    max_qty = (balance * leverage) / entry
    return round(min(qty, max_qty), 3)

# ================== INPUT USER ==================
symbols = get_all_symbols()
symbol = st.sidebar.selectbox("🔄 Pilih Pair Trading:", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
interval = st.sidebar.selectbox("⏱️ Interval (menit):", ["1", "3", "5", "15", "30", "60"], index=0)
balance = st.sidebar.number_input("💰 Modal (USDT):", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("⚙️ Leverage", 1, 100, 10)

# ================== PROSES DATA ==================
df = get_kline_data(symbol, interval)
if not df.empty:
    df = add_indicators(df)
    signal, entry_price, take_profit, stop_loss = detect_signal(df)
else:
    st.error("❌ Gagal memuat data.")
    st.stop()

# ================== PLOT CHART ==================
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
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], line=dict(color="blue"), name="EMA 5"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], line=dict(color="orange"), name="EMA 21"))
    fig.update_layout(title=f"📉 Grafik {symbol} ({interval}m)", xaxis_rangeslider_visible=False, height=500)
    return fig

st.plotly_chart(plot_chart(df), use_container_width=True)

# ================== TAMPILKAN HASIL ==================
st.subheader(f"🤖 Sinyal AI: **{signal}**")
if signal in ["LONG", "SHORT"]:
    position_size = calculate_position_size(balance, entry_price, stop_loss, leverage)
    arah = "📈 LONG (Naik)" if signal == "LONG" else "📉 SHORT (Turun)"
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Entry", f"${entry_price:.2f}")
        st.metric("✅ Take Profit", f"${take_profit:.2f}")
    with col2:
        st.metric("🛑 Stop Loss", f"${stop_loss:.2f}")
        st.metric("📦 Position Size", f"{position_size} kontrak")
    st.caption(f"(Leverage {leverage}x, Risiko 1% dari modal ${balance:.2f})")
else:
    st.info("⏳ Belum ada sinyal. AI masih menunggu setup ideal.")

# ================== RINGKASAN DATA ==================
st.markdown("### 🔍 Ringkasan Indikator")
st.dataframe(df[["close", "rsi", "ema_fast", "ema_slow", "macd"]].tail(5).round(2))
