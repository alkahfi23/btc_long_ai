# ================== IMPORT LIBRARY ==================
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("📊 AI BTC Signal Analyzer (Multi-Timeframe Strategy)")

# ================== API FUNCTIONS ==================
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
        df = pd.DataFrame(data["result"]["list"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except:
        return pd.DataFrame()

# ================== INDICATOR & SIGNAL ==================
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    return df

def detect_signal(df, min_sl_pct=0.5):
    if df.empty or len(df) < 2:
        return "NO DATA", None, None, None

    last = df.iloc[-1]
    close = last["close"]
    if close <= 0:
        return "NO DATA", None, None, None

    ema_fast = last["ema_fast"]
    ema_slow = last["ema_slow"]
    rsi = last["rsi"]
    macd = last["macd"]

    long_cond = (rsi < 70 and ema_fast > ema_slow and macd > 0)
    short_cond = (rsi > 30 and ema_fast < ema_slow and macd < 0)

    if long_cond:
        entry = close
        tp = entry * 1.02
        sl = entry * (1 - max(min_sl_pct / 100, 0.01))
        return "LONG", entry, tp, sl
    elif short_cond:
        entry = close
        tp = entry * 0.98
        sl = entry * (1 + max(min_sl_pct / 100, 0.01))
        return "SHORT", entry, tp, sl
    else:
        return "WAIT", None, None, None

def analyze_multi_timeframe(symbol, tf_trend="15", tf_entry="3", limit=100):
    df_trend = get_kline_data(symbol, tf_trend, limit)
    df_entry = get_kline_data(symbol, tf_entry, limit)
    if df_trend.empty or df_entry.empty:
        return "NO DATA", None, None, None, None

    df_trend = add_indicators(df_trend)
    df_entry = add_indicators(df_entry)

    trend = df_trend.iloc[-1]
    trend_long = trend["ema_fast"] > trend["ema_slow"] and trend["macd"] > 0
    trend_short = trend["ema_fast"] < trend["ema_slow"] and trend["macd"] < 0

    signal, entry, tp, sl = detect_signal(df_entry)

    if signal == "LONG" and trend_long:
        return "LONG", entry, tp, sl, df_entry
    elif signal == "SHORT" and trend_short:
        return "SHORT", entry, tp, sl, df_entry
    else:
        return "WAIT", None, None, None, df_entry

# ================== RISK FUNCTIONS ==================
def estimate_historical_volatility(df, window=14):
    if df.empty or len(df) < window:
        return 0.0
    returns = df["close"].pct_change()
    volatility = returns.rolling(window=window).std() * 100
    return volatility.iloc[-1] if not volatility.empty else 0.0

def estimate_margin_call_risk(entry, stop_loss, leverage, historical_volatility):
    if entry == 0 or stop_loss == 0 or leverage == 0:
        return "❌ Data tidak valid"
    stop_pct = abs(entry - stop_loss) / entry * 100
    risk_ratio = (historical_volatility / stop_pct) * leverage
    if risk_ratio < 1.0:
        return "✅ Risiko Margin Call: Rendah"
    elif risk_ratio < 3.0:
        return "⚠️ Risiko Margin Call: Sedang"
    else:
        return "🚨 Risiko Margin Call: Tinggi"

def calculate_trailing_stop(entry, volatility, direction="LONG", multiplier=2):
    offset = (volatility / 100) * entry * multiplier
    return entry - offset if direction == "LONG" else entry + offset

def check_volatility_for_trade(vol, threshold=3.0):
    return vol <= threshold

def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0, min_sl_pct=0.5):
    if entry == 0 or sl == 0:
        return 0.0
    stop_range = abs(entry - sl)
    if (stop_range / entry) * 100 < min_sl_pct:
        st.warning(f"⚠️ Stop Loss terlalu dekat ({(stop_range / entry) * 100:.2f}%). Risiko tinggi!")
        return 0.0
    risk_amount = balance * (risk_pct / 100)
    qty = risk_amount / (stop_range / entry)
    max_qty = (balance * leverage) / entry
    safe_qty = min(qty, max_qty) * 0.9
    return round(safe_qty, 3)

def predict_market_direction(df, steps=1):
    if df.empty or len(df) < 20:
        return "Tidak cukup data", 0.0
    model = ARIMA(df['close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    direction = "Naik 🚀" if forecast[-1] > df['close'].iloc[-1] else "Turun 📉"
    return direction, forecast[-1]

# ================== CHART ==================
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Candlestick"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA 5", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA 21", line=dict(color="orange")))
    fig.update_layout(title="📉 Grafik Candlestick + EMA", xaxis_rangeslider_visible=False, height=500)
    return fig

# ================== UI ==================
symbols = get_all_symbols()
symbol = st.sidebar.selectbox("🔄 Pilih Pair:", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
entry_tf = st.sidebar.selectbox("⏱️ Timeframe Entry:", ["1", "3", "5", "15", "30", "60"], index=1)
balance = st.sidebar.number_input("💰 Modal (USDT):", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("⚙️ Leverage", 1, 100, 10)

# ================== ANALISIS ==================
signal, entry_price, take_profit, stop_loss, df_plot = analyze_multi_timeframe(symbol, tf_trend="15", tf_entry=entry_tf)

st.subheader(f"🤖 Sinyal AI: **{signal}**")
if signal in ["LONG", "SHORT"]:
    hist_vol = estimate_historical_volatility(df_plot)
    trailing_stop = calculate_trailing_stop(entry_price, hist_vol, direction=signal)
    is_safe = check_volatility_for_trade(hist_vol)
    
    if not is_safe:
        st.warning("⚠️ Volatilitas terlalu tinggi, hati-hati!")
    else:
        position_size = calculate_position_size(balance, entry_price, trailing_stop, leverage)
        arah = "📈 LONG" if signal == "LONG" else "📉 SHORT"
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Entry", f"${entry_price:.4f}")
            st.metric("✅ Take Profit", f"${take_profit:.4f}")
        with col2:
            st.metric("🛑 Stop Loss", f"${trailing_stop:.4f}")
            st.metric("📦 Posisi", f"{position_size} kontrak")
        st.caption(f"Leverage {leverage}x | Modal ${balance:.2f}")
        st.caption(f"📊 Volatilitas: {hist_vol:.2f}%")
        st.warning(estimate_margin_call_risk(entry_price, trailing_stop, leverage, hist_vol))
        arah_pred, harga_pred = predict_market_direction(df_plot)
        st.info(f"🔮 Prediksi AI: {arah_pred} → ${harga_pred:.2f}")
else:
    st.info("⏳ AI menunggu setup ideal di TF kecil dan tren besar sesuai.")

# ================== CHART & TABEL ==================
if not df_plot.empty:
    st.plotly_chart(plot_chart(df_plot), use_container_width=True)
    st.markdown("### 📌 Ringkasan Indikator")
    st.dataframe(df_plot[["close", "rsi", "ema_fast", "ema_slow", "macd"]].tail(5).round(2))
