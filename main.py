import streamlit as st
from pybit.unified_trading import HTTP
import pandas as pd
import ta
from sklearn.linear_model import LogisticRegression

# ==== KONFIGURASI ====
API_KEY = 'UTXOAmlqqeRe0YFJTR'
API_SECRET = 'FBjbeyh0ktAPzmA1J7N9KNulI2DceszaTZwm'
SYMBOL = 'BTCUSDT'

session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=False)

# ==== DATA DAN MODEL ====

@st.cache_data
def get_candles(symbol, interval=1, limit=100):
    klines = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
    if "result" in data and "list" in data["result"]:
    df = pd.DataFrame(data["result"]["list"],columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"])
    # lanjutkan proses
    else:
    st.error("❌ Gagal mengambil data dari API Bybit. Cek API Key atau jaringan.")
    return pd.DataFrame()


def train_ai_model(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['ma'] = df['close'].rolling(window=10).mean()
    df['future'] = df['close'].shift(-1)
    df['target'] = (df['future'] > df['close']).astype(int)
    df.dropna(inplace=True)
    X = df[['rsi', 'macd', 'ma']]
    y = df['target']
    model = LogisticRegression().fit(X, y)
    return model

def predict_signal(df, model):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['ma'] = df['close'].rolling(window=10).mean()
    df.dropna(inplace=True)
    last = df.iloc[-1][['rsi', 'macd', 'ma']].values.reshape(1, -1)
    return model.predict(last)[0] == 1

def train_tp_sl_model(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['ma'] = df['close'].rolling(window=10).mean()
    df['future'] = df['close'].shift(-1)
    df['max_gain'] = df['close'].rolling(10).apply(lambda x: max(x / x[0] - 1) * 100, raw=False)
    df['max_loss'] = df['close'].rolling(10).apply(lambda x: min(x / x[0] - 1) * 100, raw=False)
    df.dropna(inplace=True)

    X = df[['rsi', 'macd', 'ma']]
    tp_model = LogisticRegression().fit(X, (df['max_gain'] > 2).astype(int))
    sl_model = LogisticRegression().fit(X, (df['max_loss'] < -2).astype(int))
    return tp_model, sl_model

def predict_tp_sl(df, tp_model, sl_model):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['ma'] = df['close'].rolling(window=10).mean()
    df.dropna(inplace=True)
    last = df.iloc[-1][['rsi', 'macd', 'ma']].values.reshape(1, -1)

    tp_pred = tp_model.predict(last)[0]
    sl_pred = sl_model.predict(last)[0]

    tp_pct = 3 if tp_pred else 1.5
    sl_pct = 1 if sl_pred else 2
    return tp_pct, sl_pct

# ==== POSISI & MARGIN ====

def get_account_info():
    account = session.get_wallet_balance(accountType="UNIFIED")
    usdt_balance = float(account['result']['list'][0]['totalEquity'])
    return usdt_balance

def calculate_volatility(df):
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].rolling(window=10).std().iloc[-1]
    return volatility

def calculate_safe_position_size(df, sl_pct, leverage=10, risk_pct=0.01):
    balance = get_account_info()
    volatility = calculate_volatility(df)
    risk_dollar = balance * risk_pct
    last_price = df['close'].iloc[-1]
    loss_per_btc = last_price * (sl_pct / 100)

    if loss_per_btc == 0:
        return 0.001

    max_position = risk_dollar / loss_per_btc
    leveraged_position = max_position * leverage
    return round(max(leveraged_position, 0.001), 3)

def place_long_order(symbol, qty, tp_pct, sl_pct):
    price = float(session.get_ticker(category="linear", symbol=symbol)['result']['list'][0]['lastPrice'])
    tp_price = round(price * (1 + tp_pct / 100), 2)
    sl_price = round(price * (1 - sl_pct / 100), 2)

    session.place_order(category="linear", symbol=symbol, side="Buy", order_type="Market", qty=qty)
    session.set_trading_stop(category="linear", symbol=symbol, take_profit=tp_price, stop_loss=sl_price)
    return price, tp_price, sl_price

import plotly.graph_objects as go

def plot_chart(df, signal=False):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    ))

    # MA10
    df['ma10'] = df['close'].rolling(window=10).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df['ma10'], line=dict(color='orange'), name="MA 10"))

    # Highlight sinyal
    if signal:
        fig.add_vline(x=df.index[-1], line_width=2, line_dash="dash", line_color="green", annotation_text="AI Long", annotation_position="top left")

    fig.update_layout(title="📊 BTC/USDT Chart + Sinyal AI", height=500, xaxis_rangeslider_visible=False)
    return fig


# ==== STREAMLIT UI ====
st.title("🤖 AI Bitcoin Futures Long Signal (Bybit)")
st.caption("Dengan AI untuk sinyal, TP/SL, dan pengelolaan risiko otomatis")

st.sidebar.header("⚙️ Pengaturan Risiko")
risk_pct_input = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
leverage_input = st.sidebar.slider("Leverage", 1, 100, 10)

df = get_candles(SYMBOL)
ai_model = train_ai_model(df)
tp_model, sl_model = train_tp_sl_model(df)

signal = predict_signal(df, ai_model)
tp_pct, sl_pct = predict_tp_sl(df, tp_model, sl_model)
price = float(session.get_ticker(category="linear", symbol=SYMBOL)['result']['list'][0]['lastPrice'])
qty = calculate_safe_position_size(df, sl_pct, leverage=leverage_input, risk_pct=risk_pct_input / 100)

col1, col2 = st.columns(2)
# === Tampilkan Chart ===
df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
chart = plot_chart(df, signal)
st.plotly_chart(chart, use_container_width=True)

col1.metric("📈 BTC/USDT", f"${price:.2f}")
col2.metric("🤖 AI Sinyal", "📗 Long" if signal else "📕 No Entry")

st.markdown(f"**🎯 AI TP target:** `{tp_pct}%`")
st.markdown(f"**🛡️ AI SL limit:** `{sl_pct}%`")
st.markdown(f"**📌 Ukuran posisi (AI):** `{qty} BTC`")
st.markdown(f"**⚖️ Leverage:** `{leverage_input}x` | **Risk:** `{risk_pct_input}%` dari saldo")

if st.button("🚀 Open Long (Manual)"):
    market_price, tp_price, sl_price = place_long_order(SYMBOL, qty, tp_pct, sl_pct)
    st.success(f"Opened Long {qty} BTC @ {market_price} | TP: {tp_price} | SL: {sl_price}")
