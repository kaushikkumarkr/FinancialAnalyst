# forecasting.py

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras import Input

API_KEY = "rZObIQmyDKDzuSXPpZR8YDiiMtedlcw0"



def fetch_financial_data(company_symbol):
    url = f'https://financialmodelingprep.com/api/v3/financials/income-statement/{company_symbol}?limit=120&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    return data.get("financials", None)

def extract_financial_features(financial_data):
    if financial_data:
        df = pd.DataFrame(financial_data)
        df['date'] = pd.to_datetime(df['date'])
        available_features = [col for col in df.columns if col != "date"]
        print(f"✅ Available Features: {available_features}")
        return df, available_features
    return None, []

def get_past_data(company_symbol, selected_features):
    financial_data = fetch_financial_data(company_symbol)
    if not financial_data:
        return None
    df, available_features = extract_financial_features(financial_data)
    if not all(f in available_features for f in selected_features):
        return None
    df = df[["date"] + selected_features].copy()
    df[selected_features] = df[selected_features].apply(pd.to_numeric, errors='coerce')
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df = df.sort_values("date", ascending=True)
    return df

def train_and_forecast(company_symbol, selected_features, num_years):
    num_years = int(num_years)
    seq_length = 1  
    financial_data = fetch_financial_data(company_symbol)
    if not financial_data:
        return {"error": "No financial data available"}
    df, available_features = extract_financial_features(financial_data)
    if not all(f in available_features for f in selected_features):
        return {"error": "Selected features not available"}
    df = df[["date"] + selected_features].copy()
    df[selected_features] = df[selected_features].apply(pd.to_numeric, errors='coerce')
    df=df.ffill().bfill()
    if df[selected_features].isna().all().all():
        return {"error": "No valid data to forecast"}
    df = df[::-1]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[selected_features])
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length])
    if not X:
        return {"error": "Insufficient data to create sequences"}
    X = np.array(X).reshape((-1, seq_length, len(selected_features)))
    y = np.array(y)

    model_lstm = Sequential([
        Input(shape=(seq_length, len(selected_features))),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(16, activation='relu'),
        Dense(len(selected_features))
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X, y, epochs=100, batch_size=1, verbose=0)
    history = model_lstm.fit(X, y, epochs=100, batch_size=1, verbose=0)
    print("Final LSTM loss:", history.history['loss'][-1]) 

    lstm_preds = []
    seq = X[-1]
    for _ in range(num_years):
        pred = model_lstm.predict(seq.reshape((1, seq_length, len(selected_features))))[0]
        lstm_preds.append(pred)
        seq = np.append(seq[1:], pred).reshape((1, seq_length, len(selected_features)))
    lstm_preds = scaler.inverse_transform(lstm_preds)

    arima_preds, holt_preds = [], []
    for f in selected_features:
        s = df[f].astype(float)
        try:
            arima_model = ARIMA(s, order=(1, 1, 0)).fit()
            arima_preds.append(arima_model.forecast(num_years).tolist())
        except:
            arima_preds.append([0] * num_years)
        try:
            hw_model = ExponentialSmoothing(s, seasonal="add", seasonal_periods=4).fit()
            holt_preds.append(hw_model.forecast(num_years).tolist())
        except:
            holt_preds.append([0] * num_years)
    arima_preds = np.array(arima_preds).T
    holt_preds = np.array(holt_preds).T

    final = 0.5 * lstm_preds + 0.3 * arima_preds + 0.2 * holt_preds
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=num_years+1, freq='YE')[1:]
    future_df = pd.DataFrame(final, columns=selected_features, index=future_dates)
    return future_df.reset_index().rename(columns={"index": "date"})
