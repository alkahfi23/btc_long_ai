import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import plotly.graph_objects as go
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

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

def detect_market_regime(df):
    adx = df["adx"]
    ema_fast = df["ema_fast"]
    ema_slow = df["ema_slow"]
    trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1] and adx.iloc[-1] > 20
    trend_down = ema_fast.iloc[-1] < ema_slow.iloc[-1] and adx.iloc[-1] > 20
    if trend_up:
        return "UPTREND"
    elif trend_down:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

def calculate_position_size(entry_price, stop_loss, modal, risk_pct, leverage):
    risk_amount = modal * (risk_pct / 100)
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        return 0
    qty = (risk_amount / risk_per_unit) * leverage
    return round(qty, 3)

def build_and_train_lstm(df):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['close']])
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60, 1)),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    model.save("lstm_model.h5")
    st.success("✅ Model LSTM berhasil dilatih dan disimpan!")

def load_lstm_model():
    try:
        model = load_model("lstm_model.h5")
        return model
    except:
        st.warning("Model LSTM tidak ditemukan. Silakan latih ulang terlebih dahulu.")
        return None

def predict_lstm(df, model):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['close']])
    X = np.array([data_scaled[-60:]])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    pred = model.predict(X, verbose=0)
    pred_price = scaler.inverse_transform(pred)[0][0]
    return pred_price

def plot_chart(df, signal, tp=None, sl=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], line=dict(color='blue', width=1), name='EMA 5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], line=dict(color='orange', width=1), name='EMA 20'))

    if tp:
        fig.add_shape(type="line",
                      x0=df.index[0], x1=df.index[-1],
                      y0=tp, y1=tp,
                      line=dict(color="green", dash="dash"),
                      name="TP")
        fig.add_annotation(x=df.index[-1], y=tp, text="TP", showarrow=False, yshift=10, font=dict(color="green"))

    if sl:
        fig.add_shape(type="line",
                      x0=df.index[0], x1=df.index[-1],
                      y0=sl, y1=sl,
                      line=dict(color="red", dash="dash"),
                      name="SL")
        fig.add_annotation(x=df.index[-1], y=sl, text="SL", showarrow=False, yshift=-10, font=dict(color="red"))

    fig.update_layout(title=f"Sinyal: {signal}", xaxis_title='Waktu', yaxis_title='Harga')
    return fig

def detect_signal(df):
    last = df.iloc[-1]
    regime = detect_market_regime(df)
    entry = last["close"]
    model = load_lstm_model()
    lstm_pred = predict_lstm(df, model) if model else entry

    if regime in ["UPTREND", "DOWNTREND"]:
        signal = detect_trend_signal(df, last, entry, lstm_pred)
    elif regime == "SIDEWAYS":
        signal = detect_sideways_signal(df, last, entry, lstm_pred)
    else:
        signal = "WAIT", None, None, lstm_pred > entry

    return signal, entry, lstm_pred

for symbol in ["BTCUSDT", "ETHUSDT"]:
    df = get_kline_data(symbol, interval=interval, csv_df=df_csv if mode_data == "Backtest" else None)
    if not df.empty:
        signal, entry, lstm_pred = detect_signal(df)
        if signal_filter == "SEMUA" or signal == signal_filter:
                        st.subheader(f"Sinyal untuk {symbol}")
            st.write(f"Regim Pasar: {detect_market_regime(df)}")
            st.write(f"Prediksi LSTM: {lstm_pred:.2f}")
            
            # Menampilkan sinyal
            if signal == "LONG":
                st.write("📈 Sinyal: LONG")
            elif signal == "SHORT":
                st.write("📉 Sinyal: SHORT")
            elif signal == "WAIT":
                st.write("⏸️ Sinyal: WAIT")
            else:
                st.write("❓ Sinyal: Tidak Terdefinisi")

            # Menghitung TP dan SL
            tp = None
            sl = None
            if signal == "LONG":
                tp = entry + (entry * 0.02)  # Take Profit 2% lebih tinggi dari entry
                sl = entry - (entry * 0.01)  # Stop Loss 1% lebih rendah dari entry
            elif signal == "SHORT":
                tp = entry - (entry * 0.02)  # Take Profit 2% lebih rendah dari entry
                sl = entry + (entry * 0.01)  # Stop Loss 1% lebih tinggi dari entry

            # Visualisasi chart
            fig = plot_chart(df, signal, tp, sl)
            st.plotly_chart(fig)

            # Menampilkan informasi posisi
            qty = calculate_position_size(entry, sl, modal, risk_pct, leverage)
            st.write(f"🧮 Ukuran Posisi: {qty} {symbol.split('USDT')[0]}")
            st.write(f"📉 Entry Price: {entry:.2f} USD")
            st.write(f"🎯 Take Profit: {tp:.2f} USD")
            st.write(f"🚨 Stop Loss: {sl:.2f} USD")
            st.write(f"💵 Modal: ${modal}")

        st.session_state.analyzed = True

else:
    st.warning("Data tidak tersedia untuk analisis. Silakan pilih data atau upload file CSV untuk backtesting.")

