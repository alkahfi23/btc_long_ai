import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta

st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("\ud83d\udcca AI BTC Signal Analyzer (Multi-Timeframe Strategy)")

# ================== API FUNCTIONS ==================
@st.cache_data(ttl=3600)
def get_all_symbols():
    url = "https://api.bybit.com/v5/market/instruments-info"
    try:
        res = requests.get(url, params={"category": "linear"}, timeout=10)
        data = res.json()
        return sorted([i["symbol"] for i in data["result"]["list"] if "USDT" in i["symbol"]])
    except:
        return ["symbol"]

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
    df["volume_avg"] = df["volume"].rolling(window=14).mean()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

def compute_signal_confidence(df):
    last = df.iloc[-1]
    score = 0
    if last["rsi"] > 50: score += 1
    if last["ema_fast"] > last["ema_slow"]: score += 1
    if last["macd"] > 0: score += 1
    if last["volume"] > df["volume_avg"].iloc[-1]: score += 1
    return round((score / 4) * 100, 2)

def detect_signal(df):
    if df.empty or len(df) < 3:
        return "NO DATA", None, None, None, 0.0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    long_cond = (
        last["rsi"] > prev["rsi"] > 50 and
        last["ema_fast"] > last["ema_slow"] and
        last["macd"] > 0 and
        last["volume"] > df["volume_avg"].iloc[-1]
    )
    short_cond = (
        last["rsi"] < prev["rsi"] < 50 and
        last["ema_fast"] < last["ema_slow"] and
        last["macd"] < 0 and
        last["volume"] > df["volume_avg"].iloc[-1]
    )
    entry = last["close"]
    atr = last["atr"]
    if long_cond:
        return "LONG", entry, entry + 2 * atr, entry - atr, compute_signal_confidence(df)
    elif short_cond:
        return "SHORT", entry, entry - 2 * atr, entry + atr, compute_signal_confidence(df)
    else:
        return "WAIT", None, None, None, compute_signal_confidence(df)

def is_strong_trend(df, direction="long", candles=3):
    if len(df) < candles:
        return False
    recent = df[-candles:]
    if direction == "long":
        return all(r["ema_fast"] > r["ema_slow"] and r["macd"] > 0 for _, r in recent.iterrows())
    else:
        return all(r["ema_fast"] < r["ema_slow"] and r["macd"] < 0 for _, r in recent.iterrows())

def analyze_multi_timeframe(symbol, tf_trend="15", tf_entry="3", limit=100):
    df_trend = get_kline_data(symbol, tf_trend, limit)
    df_entry = get_kline_data(symbol, tf_entry, limit)
    if df_trend.empty or df_entry.empty:
        return "NO DATA", None, None, None, None, 0.0

    df_trend = add_indicators(df_trend)
    df_entry = add_indicators(df_entry)

    trend_long = is_strong_trend(df_trend, "long")
    trend_short = is_strong_trend(df_trend, "short")

    signal, entry, tp, sl, confidence = detect_signal(df_entry)
    if signal == "LONG" and trend_long:
        return "LONG", entry, tp, sl, df_entry, confidence
    elif signal == "SHORT" and trend_short:
        return "SHORT", entry, tp, sl, df_entry, confidence
    else:
        return "WAIT", None, None, None, df_entry, confidence

# ================== VOLATILITY & MARGIN RISK ==================
def estimate_historical_volatility(df, window=14):
    if df.empty or len(df) < window:
        return 0.0
    returns = df["close"].pct_change()
    volatility = returns.rolling(window=window).std() * 100
    return volatility.iloc[-1] if not volatility.empty else 0.0

def estimate_margin_call_risk(entry, stop_loss, leverage, historical_volatility):
    if entry == 0 or stop_loss == 0 or leverage == 0:
        return "\u274c Data tidak valid"
    stop_pct = abs(entry - stop_loss) / entry * 100
    risk_ratio = (historical_volatility / stop_pct) * leverage
    if risk_ratio < 1.0:
        return "\u2705 Risiko Margin Call: Rendah"
    elif risk_ratio < 3.0:
        return "\u26a0\ufe0f Risiko Margin Call: Sedang"
    else:
        return "\ud83d\udea8 Risiko Margin Call: Tinggi"

# ================== POSISI AMAN ==================
def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0, min_sl_pct=0.5):
    if entry == 0 or sl == 0:
        return 0.0
    stop_range = abs(entry - sl)
    if (stop_range / entry) * 100 < min_sl_pct:
        st.warning(f"\u26a0\ufe0f Stop Loss terlalu dekat ({(stop_range / entry) * 100:.2f}%). Risiko terlalu tinggi!")
        return 0.0
    risk_amount = balance * (risk_pct / 100)
    qty = risk_amount / (stop_range / entry)
    max_qty = (balance * leverage) / entry
    safe_qty = min(qty, max_qty) * 0.9
    return round(safe_qty, 3)

# ================== CHART ==================
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA 5", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA 21", line=dict(color="orange")))
    fig.update_layout(title="\ud83d\udcc9 Grafik Candlestick + EMA", xaxis_rangeslider_visible=False, height=500)
    return fig

# ================== UI ==================
symbols = get_all_symbols()
symbol = st.sidebar.selectbox("\ud83d\udd04 Pilih Pair:", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
entry_tf = st.sidebar.selectbox("\u23f1\ufe0f Timeframe Entry:", ["1", "3", "5", "15", "30", "60"], index=1)
balance = st.sidebar.number_input("\ud83d\udcb0 Modal (USDT):", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("\u2699\ufe0f Leverage", 1, 100, 10)

signal, entry_price, take_profit, stop_loss, df_plot, confidence = analyze_multi_timeframe(symbol, tf_trend="15", tf_entry=entry_tf)

st.subheader(f"\ud83e\udd16 Sinyal AI (Multi-Timeframe): **{signal}**")
if signal in ["LONG", "SHORT"]:
    position_size = calculate_position_size(balance, entry_price, stop_loss, leverage)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("\ud83c\udfaf Entry", f"${entry_price:.4f}")
        st.metric("\u2705 Take Profit", f"${take_profit:.4f}")
    with col2:
        st.metric("\ud83d\udea9 Stop Loss", f"${stop_loss:.4f}")
        st.metric("\ud83d\udce6 Posisi", f"{position_size} kontrak")
    st.caption(f"(Leverage {leverage}x | Modal ${balance:.2f})")
    st.caption(f"\ud83c\udf1f Kepercayaan Sinyal: {confidence:.1f}%")
    hist_vol = estimate_historical_volatility(df_plot)
    st.caption(f"\ud83d\udcc8 Estimasi Volatilitas: {hist_vol:.2f}%")
    risk_warning = estimate_margin_call_risk(entry_price, stop_loss, leverage, hist_vol)
    st.warning(risk_warning)
else:
    st.info("\u23f3 AI menunggu setup ideal di TF kecil *dan* arah tren besar yang sesuai.")

if not df_plot.empty:
    st.plotly_chart(plot_chart(df_plot), use_container_width=True)
    st.markdown("### \ud83d\udccc Ringkasan Indikator")
    st.dataframe(df_plot[["close", "rsi", "ema_fast", "ema_slow", "macd"]].tail(5).round(2))
