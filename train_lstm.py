import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib
import argparse
import os

def fetch_data(symbol="BTCUSDT", interval="60", limit=1000):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()["result"]["list"]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def load_csv_data(filepath):
    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return pd.DataFrame()

def preprocess_data(df, seq_length=60):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[["close"]])
    
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_and_save(df, symbol="CUSTOM", save_dir="."):
    if df.empty:
        print("❌ Data kosong. Gagal melatih model.")
        return

    X, y, scaler = preprocess_data(df)
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    model_filename = os.path.join(save_dir, f"lstm_model_{symbol}.h5")
    scaler_filename = os.path.join(save_dir, f"scaler_{symbol}.pkl")
    
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"✅ Model disimpan: {model_filename}")
    print(f"✅ Scaler disimpan: {scaler_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Nama simbol (untuk penamaan file model)")
    parser.add_argument("--interval", type=str, default="60", help="Interval Bybit (misalnya 60)")
    parser.add_argument("--csv", type=str, default=None, help="Path ke file CSV (jika ingin custom training)")
    args = parser.parse_args()

    if args.csv:
        df = load_csv_data(args.csv)
        symbol_name = os.path.splitext(os.path.basename(args.csv))[0]
        train_and_save(df, symbol=symbol_name)
    else:
        df = fetch_data(args.symbol, args.interval)
        train_and_save(df, symbol=args.symbol)
