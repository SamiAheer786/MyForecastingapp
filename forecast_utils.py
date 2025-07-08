import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from numpy import polyfit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from calendar import monthrange

def preprocess_data(df, date_col, target_col, filters=[]):
    df = df[[date_col, target_col] + filters].copy()
    df.columns = ['date', 'target'] + filters
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['date', 'target'], inplace=True)
    return df

def forecast_sales(df, model_type, forecast_until='year_end', custom_days=None, grouping_option="Daily"):
    df['ds'] = df['date']
    if grouping_option == "Monthly":
        df['ds'] = df['ds'].dt.to_period('M').dt.to_timestamp()
    elif grouping_option == "Yearly":
        df['ds'] = df['ds'].dt.to_period('Y').dt.to_timestamp()

    df_grouped = df.groupby("ds")['target'].sum().reset_index().sort_values("ds")
    df_grouped.columns = ['ds', 'y']
    last_data_date = df_grouped['ds'].max()

    if forecast_until == 'month_end':
        end_date = datetime(last_data_date.year, last_data_date.month, monthrange(last_data_date.year, last_data_date.month)[1])
    elif forecast_until == 'quarter_end':
        q_month = ((last_data_date.month - 1) // 3 + 1) * 3
        end_date = datetime(last_data_date.year, q_month, monthrange(last_data_date.year, q_month)[1])
    elif forecast_until == 'custom':
        end_date = last_data_date + timedelta(days=custom_days)
    else:
        end_date = datetime(last_data_date.year, 12, 31)

    if grouping_option == "Monthly":
        future_dates = pd.date_range(start=last_data_date + pd.offsets.MonthBegin(), end=end_date, freq='MS')
    elif grouping_option == "Yearly":
        future_dates = pd.date_range(start=last_data_date + pd.offsets.YearBegin(), end=end_date, freq='YS')
    else:
        future_dates = pd.date_range(start=last_data_date + timedelta(days=1), end=end_date)

    if model_type == "Prophet":
        model = Prophet()
        model.fit(df_grouped)
        future = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future)[['ds', 'yhat']]
    elif model_type == "Linear":
        df_grouped['ds_ord'] = df_grouped['ds'].map(datetime.toordinal)
        m, b = polyfit(df_grouped['ds_ord'], df_grouped['y'], 1)
        forecast = pd.DataFrame({'ds': future_dates})
        forecast['yhat'] = [m * d.toordinal() + b for d in forecast['ds']]
    elif model_type == "Exponential":
        model = ExponentialSmoothing(df_grouped['y'], trend='add').fit()
        forecast_vals = model.forecast(len(future_dates))
        forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_vals})
    else:
        return pd.DataFrame(), last_data_date, 0, df_grouped

    forecast_full = pd.concat([
        df_grouped.rename(columns={'y': 'yhat'}),
        forecast
    ]).sort_values('ds')

    return forecast, last_data_date, len(future_dates), forecast_full

def get_forecast_explanation(method):
    return {
        "Prophet": "Prophet models trends and seasonality.",
        "Linear": "Linear regression fits a trend line.",
        "Exponential": "Exponential smoothing emphasizes recent data."
    }.get(method, "")
