import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="AI BTC Long Signal", layout="wide")
st.title("🚀 AI BTC Long Signal Analyzer")

# Dummy data (replace with real API and AI logic)
df = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
    'open': pd.Series(range(100)) + 30000,
    'high': pd.Series(range(100)) + 30100,
    'low': pd.Series(range(100)) + 29900,
    'close': pd.Series(range(100)) + 30050,
})

df.set_index('timestamp', inplace=True)

# Chart function
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    ))
    df['ma10'] = df['close'].rolling(window=10).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df['ma10'], line=dict(color='orange'), name="MA 10"))
    fig.update_layout(title="📊 BTC/USDT Chart", height=500, xaxis_rangeslider_visible=False)
    return fig

st.plotly_chart(plot_chart(df), use_container_width=True)
st.success("✅ Sinyal AI: LONG (simulasi)")