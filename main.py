import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# ==== Tambahan untuk LSTM ====
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ============ CONFIG & TITLE ============
st.set_page_config(page_title="AI BTC Signal Analyzer", layout="wide")
st.title("📊 AI BTC Signal Analyzer (Multi-Timeframe Strategy)")

# ============ API FUNCTIONS ============
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

# ============ INDICATORS & SIGNAL ============
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

# ============ VOLATILITY & RISK ============
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

# ============ TRAILING STOP ============
def calculate_trailing_stop(entry, volatility, direction="LONG", multiplier=2):
    stop_loss = entry - volatility * multiplier if direction == "LONG" else entry + volatility * multiplier
    return stop_loss

def check_volatility_for_trade(volatility, threshold=3.0):
    return volatility <= threshold

# ============ POSITION SIZE ============
def calculate_position_size(balance, entry, sl, leverage=10, risk_pct=1.0, min_sl_pct=0.5):
    if entry == 0 or sl == 0:
        return 0.0
    stop_range = abs(entry - sl)
    if (stop_range / entry) * 100 < min_sl_pct:
        st.warning(f"⚠️ Stop Loss terlalu dekat ({(stop_range / entry) * 100:.2f}%). Risiko terlalu tinggi!")
        return 0.0
    risk_amount = balance * (risk_pct / 100)
    qty = risk_amount / (stop_range / entry)
    max_qty = (balance * leverage) / entry
    safe_qty = min(qty, max_qty) * 0.9
    return round(safe_qty, 3)

# ============ CHART ============
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
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA 5", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA 21", line=dict(color="orange")))
    fig.update_layout(title="📉 Grafik Candlestick + EMA", xaxis_rangeslider_visible=False, height=500)
    return fig

# ============ PREDIKSI LSTM ============
def prepare_lstm_data(df, lookback=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']])
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_lstm_model(X, y):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    model.fit(X, y, epochs=5, batch_size=32, verbose=0, callbacks=[es])
    return model

@st.cache_resource(show_spinner=False)
def cached_lstm_model(df, lookback=30):
    X, y, scaler = prepare_lstm_data(df, lookback)
    model = train_lstm_model(X, y)
    return model, scaler

def predict_with_lstm(df, lookback=30):
    if len(df) < lookback + 1:
        return "Tidak cukup data", 0.0
    df = df.copy()
    model, scaler = cached_lstm_model(df, lookback)
    last_sequence = df[['close']].values[-lookback:]
    last_scaled = scaler.transform(last_sequence)
    last_input = np.reshape(last_scaled, (1, lookback, 1))
    prediction_scaled = model.predict(last_input, verbose=0)
    predicted_price = scaler.inverse_transform(prediction_scaled)[0][0]
    direction = "Naik 🚀" if predicted_price > df['close'].iloc[-1] else "Turun 📉"
    return direction, predicted_price

# ============ UI ============
symbols = get_all_symbols()
symbol = st.sidebar.selectbox("🔄 Pilih Pair:", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
entry_tf = st.sidebar.selectbox("⏱️ Timeframe Entry:", ["1", "3", "5", "15", "30", "60"], index=1)
balance = st.sidebar.number_input("💰 Modal (USDT):", min_value=10.0, value=100.0)
leverage = st.sidebar.slider("⚙️ Leverage", 1, 100, 10)

# ============ ANALISA ============
signal, entry_price, take_profit, stop_loss, df_plot = analyze_multi_timeframe(symbol, tf_trend="15", tf_entry=entry_tf)

st.subheader(f"🤖 Sinyal AI (Multi-Timeframe): **{signal}**")
if signal in ["LONG", "SHORT"]:
    hist_vol = estimate_historical_volatility(df_plot)
    trailing_stop = calculate_trailing_stop(entry_price, hist_vol)
    is_volatility_safe = check_volatility_for_trade(hist_vol)

    if not is_volatility_safe:
        st.warning("⚠️ Volatilitas pasar terlalu tinggi, berhati-hatilah!")
    else:
        position_size = calculate_position_size(balance, entry_price, trailing_stop, leverage)
        arah = "📈 LONG (Naik)" if signal == "LONG" else "📉 SHORT (Turun)"
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Entry", f"${entry_price:.4f}")
            st.metric("✅ Take Profit", f"${take_profit:.4f}")
        with col2:
            st.metric("🛑 Trailing Stop Loss", f"${trailing_stop:.4f}")
            st.metric("📦 Posisi", f"{position_size} kontrak")
        st.caption(f"(Leverage {leverage}x | Modal ${balance:.2f})")
        st.caption(f"📈 Estimasi Volatilitas: {hist_vol:.2f}%")
        risk_warning = estimate_margin_call_risk(entry_price, trailing_stop, leverage, hist_vol)
        st.warning(risk_warning)
else:
    st.info("⏳ AI menunggu setup ideal di TF kecil *dan* arah tren besar yang sesuai.")

# ============ GRAFIK ============
if not df_plot.empty:
    st.plotly_chart(plot_chart(df_plot), use_container_width=True)
    st.markdown("### 📌 Ringkasan Indikator")
    st.dataframe(df_plot[["close", "rsi", "ema_fast", "ema_slow", "macd"]].tail(5).round(2))

# ============ LSTM PREDICTION ============
with st.expander("🔮 Prediksi AI (LSTM)"):
    with st.spinner("Melatih model LSTM..."):
        direction, predicted_price = predict_with_lstm(df_plot)
        st.metric("📌 Arah Prediksi", direction)
        st.metric("💡 Harga Diprediksi", f"${predicted_price:.2f}")
