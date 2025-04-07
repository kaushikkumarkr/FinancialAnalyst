import pandas as pd
from prophet import Prophet

def forecast_kpi(data: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    """
    data: pd.DataFrame with columns ['ds', 'y']
        - ds: datetime column
        - y: value (e.g., revenue or income)
    periods: how many future years to forecast
    """
    model = Prophet(yearly_seasonality=True)
    model.fit(data)

    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
