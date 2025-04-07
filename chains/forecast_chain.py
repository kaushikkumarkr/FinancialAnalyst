# # chains/forecast_chain.py

# from tools.yahoo_api import get_financial_metric
# from tools.forecast_tool import forecast_kpi
# from tools.llm_metric_normalizer import normalize_metric_name  # optional LLM-powered helper
# import pandas as pd
# from langchain_core.language_models.base import BaseLanguageModel
# import yfinance as yf
# from tools.forecasting import train_and_forecast
# from tools.feature_mapping import validate_feature_with_llm, available_features

# class ForecastChain:
#     def __init__(self, llm: BaseLanguageModel):
#         self.llm = llm

#     def _extract_llm_text(self, response) -> str:
#         """Extract plain text from LLM response, ignoring any <think> tags or explanations."""
#         import re
#         if hasattr(response, "content"):
#             content = response.content
#         elif isinstance(response, str):
#             content = response
#         else:
#             content = str(response)

#     # Match only the last quoted string or last valid line
#         match = re.search(r"(?i)(total revenue|operating revenue|revenue|none)", content, re.IGNORECASE)
#         return match.group(1).title() if match else "None"


#     def forecast(self, company: str, ticker: str, metric: str, periods: int = 3) -> str:
#         try:
#             # Step 1: Pull available financial metrics using yfinance
#             stock = yf.Ticker(ticker)
#             fin = stock.financials.transpose()
#             if fin.empty:
#                 return f"❌ No financial data found for {ticker}."
    
#             available_metrics = list(fin.columns)
#             metric_title = metric.title()
    
#             # Step 2: Normalize metric
#             if metric_title not in available_metrics:
#                 prompt = f"""
#                 A user wants to forecast '{metric}' for {company}.
#                 Choose the best matching metric from this list:
    
#                 {available_metrics}
    
#                 Return the exact best matching name, or "None" if not found.
#                 """
#                 raw_response = self.llm.invoke(prompt.strip())
#                 metric_key = self._extract_llm_text(raw_response)
    
#                 if metric_key == "None":
#                     return f"❌ Metric '{metric}' could not be resolved for {ticker}."
#             else:
#                 metric_key = metric_title
    
#             # Step 3: Retrieve data and forecast
#             df = get_financial_metric(ticker, metric_key)
#             if df.empty or df.shape[0] < 2:
#                 return f"❌ Not enough data to forecast {metric_key} for {company}."
    
#             forecast_df = forecast_kpi(df, periods=periods)
#             summary = f"📈 Forecast for {company}'s {metric_key} (next {periods} years):\n"
#             for _, row in forecast_df.tail(periods).iterrows():
#                 summary += f"- {row['ds'].year}: ${row['yhat'] / 1e9:.2f}B\n"
    
#             return summary
    
#         except Exception as e:
#             return f"❌ Forecasting failed: {e}"

# # chains/forecast_chain.py














import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras import Input
from prophet import Prophet
import yfinance as yf  # Yahoo Finance API

class ForecastChain:
    def __init__(self, llm):
        self.llm = llm

    def get_past_data(self, ticker: str, metric: str):
        try:
            # Fetching historical data using Yahoo Finance for Income Statement
            ticker_obj = yf.Ticker(ticker)
            income_statement = ticker_obj.financials.T

            if metric not in income_statement.columns:
                available_metrics = income_statement.columns.tolist()
                return None, f"❌ Metric '{metric}' not found. Available metrics: {available_metrics}"

        # Prepare data for forecasting
            df = income_statement[[metric]].reset_index()
            df.columns = ['date', 'y']
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
            if df.empty or df.isna().all().all():
                return None, "❌ No valid data to forecast."
            print("📊 Data to be used for forecasting:")
            print(df.head())
            return df, None
        except Exception as e:
            return None, f"❌ Error fetching data: {str(e)}"


    def train_and_forecast(self, ticker, selected_features, num_years):
        num_years = int(num_years)
        seq_length = 1  
        df, error = self.get_past_data(ticker, selected_features[0])
        if error:
            return {"error": error}

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['y']])
        X, y = [], []
        for i in range(len(scaled) - seq_length):
            X.append(scaled[i:i+seq_length])
            y.append(scaled[i+seq_length])
        if not X:
            return {"error": "Insufficient data to create sequences"}
        X = np.array(X).reshape((-1, seq_length, 1))
        y = np.array(y)

        # LSTM Model
        model_lstm = Sequential([
            Input(shape=(seq_length, 1)),
            LSTM(64, activation='relu', return_sequences=True),
            LSTM(32, activation='relu', return_sequences=True),
            LSTM(16, activation='relu'),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X, y, epochs=20, batch_size=1, verbose=0)

        lstm_preds = []
        seq = X[-1]
        for _ in range(num_years):
            pred = model_lstm.predict(seq.reshape((1, seq_length, 1)))[0]
            lstm_preds.append(pred)
            seq = np.append(seq[1:], pred).reshape((1, seq_length, 1))
        lstm_preds = scaler.inverse_transform(lstm_preds)

        # ARIMA & Holt-Winters Models
        arima_preds, holt_preds = [], []
        s = df['y'].astype(float)
        try:
            arima_model = ARIMA(s, order=(1, 1, 0)).fit()
            arima_preds = arima_model.forecast(num_years).tolist()
        except:
            arima_preds = [0] * num_years

        try:
            hw_model = ExponentialSmoothing(s, seasonal="add", seasonal_periods=4).fit()
            holt_preds = hw_model.forecast(num_years).tolist()
        except:
            holt_preds = [0] * num_years

        # Prophet Model
        prophet_data = df.rename(columns={'date': 'ds', 'y': 'y'})
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=num_years, freq='Y')
        prophet_forecast = model.predict(future)
        print("📈 Prophet Forecast Sample:")
        print(prophet_forecast.tail())
        prophet_preds = prophet_forecast['yhat'].tail(num_years).to_numpy()
        prophet_preds = np.maximum(prophet_preds, 0)

        lstm_preds = np.nan_to_num(scaler.inverse_transform(lstm_preds))
        arima_preds = np.nan_to_num(arima_preds)
        holt_preds = np.nan_to_num(holt_preds)
        prophet_preds = np.nan_to_num(prophet_preds)


        # Combine all predictions
        final_preds = 0.4 * np.array(lstm_preds).flatten() + \
              0.2 * np.array(arima_preds) + \
              0.2 * np.array(holt_preds) + \
              0.2 * np.array(prophet_preds)
        final_preds = np.maximum(final_preds, 0)
        print("📈 LSTM Predictions:", lstm_preds)
        print("📈 ARIMA Predictions:", arima_preds)
        print("📈 Holt-Winters Predictions:", holt_preds)
        print("📈 Prophet Predictions:", prophet_preds)

        last_date = df["date"].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=num_years+1, freq='Y')[1:]
        future_df = pd.DataFrame(final_preds, columns=['Forecasted Revenue'], index=future_dates)
        return future_df.reset_index().rename(columns={"index": "date"})
    
    def forecast(self, company, ticker, metric, periods):
        past_data, error = self.get_past_data(ticker, metric)
        
        if past_data is None:
            return {"error": error}

        # Perform hybrid forecasting
        hybrid_forecast = self.train_and_forecast(ticker, [metric], periods)
        if isinstance(hybrid_forecast, dict) and 'error' in hybrid_forecast:
            return hybrid_forecast

        return hybrid_forecast
