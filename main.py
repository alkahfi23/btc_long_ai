import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta

st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("📊 AI BTC Signal Analyzer (Multi-Timeframe Strategy)")

# ================== API FUNCTIONS ==================
@st.cache_resource(ttl=3600)
def get_all_symbols():
    url = "https://api.bybit.com/v5/market/instruments-info"
    try:
        res = requests.get(url, params={"category": "linear"}, timeout=10)
        data = res.json()
        return sorted([i["symbol"] for i in data["result"]["list"] if "USDT" in i["symbol"]])
    except:
        return ["BTCUSDT"]

@st.cache_resource(ttl=60)
def get_kline_data(symbol, interval="1", limit=100):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
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

# ================== ANALYSIS ==================
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()

    # Menambahkan indikator Balance of Power (BOP)
    df["bop"] = ta.volume.BalanceOfPowerIndicator(df["close"], df["volume"]).balance_of_power()
    
    return df

def detect_signal(df):
    if df.empty:
        return "NO DATA", None, None, None
    last = df.iloc[-1]
    long_cond = (last["rsi"] < 70 and last["ema_fast"] > last["ema_slow"] and last["macd"] > 0)
    short_cond = (last["rsi"] > 30 and last["ema_fast"] < last["ema_slow"] and last["macd"] < 0)
    if long_cond:
        entry = last["close"]
        return "LONG", entry, entry * 1.02, entry * 0.99
    elif short_cond:
        entry = last["close"]
        return "SHORT", entry, entry * 0.98, entry * 1.01
    else:
        return "WAIT", None, None, None

def analyze_multi_timeframe(symbol, tf_trend="15", tf_entry="3", limit=100):
    df_trend = get_kline_data(symbol, tf_trend, limit)
    df_entry = get_kline_data(symbol, tf_entry, limit)
    if df_trend.empty or df_entry.empty:
        return "NO DATA", None, None, None

    df_trend = add_indicators(df_trend)
    df_entry = add_indicators(df_entry)

    # Validasi tren
    trend = df_trend.iloc[-1]
    trend_long = trend["ema_fast"] > trend["ema_slow"] and trend["macd"] > 0
    trend_short = trend["ema_fast"] < trend["ema_slow"] and trend["macd"] < 0

    signal, entry, tp, sl = detect_signal(df_entry)
    if signal == "LONG" and trend_long:
        return "LONG", entry, tp, sl
    elif signal == "SHORT" and trend_short:
        return "SHORT", entry, tp, sl
    else:
        return "WAIT", None, None, None

def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0):
    risk_amount = balance * (risk_pct / 100)
    stop_range = abs(entry - sl)
    qty = risk_amount / (stop_range / entry)
    max_qty = (balance * leverage) / entry
    return round(min(qty, max_qty), 3)

# ================== INPUT SIDEBAR ==================
symbols = get_all_symbols()
symbol = st.sidebar.selectbox("🔄 Pilih Pair:", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
entry_tf = st.sidebar.selectbox("⏱️ Timeframe Entry:", ["1", "3", "5", "15", "30", "60"], index=1)
balance = st.sidebar.number_input("💰 Modal (USDT):", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("⚙️ Leverage", 1, 100, 10)

# ================== PROSES SINYAL MTA ==================
signal, entry_price, take_profit, stop_loss = analyze_multi_timeframe(symbol, tf_trend="15", tf_entry=entry_tf)

# ================== PLOT ==================
def plot_chart(df):
    fig = go.Figure()

    # Candlestick utama
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Candlestick"
    ))

    # EMA Fast dan Slow
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA 5", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA 21", line=dict(color="orange")))

    # Layout atas
    fig.update_layout(title="📉 Candlestick + EMA", xaxis_rangeslider_visible=False, height=500)

    # Tambah subplot BOP
    bop_trace = go.Bar(x=df.index, y=df["bop"], name="Balance of Power", marker_color="purple", opacity=0.5, yaxis="y2")

    # Tambahkan axis kedua untuk BOP
    fig.update_layout(
        yaxis=dict(domain=[0.3, 1]),
        yaxis2=dict(domain=[0, 0.25], title="BOP", showgrid=True),
        height=600
    )
    fig.add_trace(bop_trace)

    return fig

# ================== TAMPILKAN HASIL ==================
st.subheader(f"🤖 Sinyal AI (Multi-Timeframe): **{signal}**")
if signal in ["LONG", "SHORT"]:
    position_size = calculate_position_size(balance, entry_price, stop_loss, leverage)
    arah = "📈 LONG (Naik)" if signal == "LONG" else "📉 SHORT (Turun)"
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Entry", f"${entry_price:.2f}")
        st.metric("✅ Take Profit", f"${take_profit:.2f}")
    with col2:
        st.metric("🛑 Stop Loss", f"${stop_loss:.2f}")
        st.metric("📦 Posisi", f"{position_size} kontrak")
    st.caption(f"(Leverage {leverage}x | Modal ${balance:.2f})")
else:
    st.info("⏳ AI menunggu setup ideal di TF kecil *dan* arah tren besar yang sesuai.")

# ================== RINGKASAN ==================
df_plot = get_kline_data(symbol, entry_tf, limit=100)
df_plot = add_indicators(df_plot)
st.markdown("### 🔍 Ringkasan Indikator")
st.dataframe(df_plot[["close", "rsi", "ema_fast", "ema_slow", "macd", "bop"]].tail(5).round(2))
