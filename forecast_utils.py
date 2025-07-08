import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from numpy import polyfit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from calendar import monthrange

def smart_date_conversion(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'month' in df.columns and 'year' in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1), errors='coerce')
    else:
        raise ValueError("No valid date or month/year columns found.")
    return df

def preprocess_data(df, date_col, target_col, filters=[]):
    df = df[[date_col, target_col] + filters].copy()
    df.columns = ['date', 'target'] + filters
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['date', 'target'], inplace=True)
    df = smart_date_conversion(df)
    return df

def forecast_sales(df, model_type, target_mode, event_dates=None, forecast_until='year_end', custom_days=None):
    df_grouped = df.groupby("date")["target"].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped = df_grouped.sort_values("ds")

    last_data_date = pd.to_datetime(df_grouped['ds'].max())

    # Determine forecast end date
    if forecast_until == 'month_end':
        year, month = last_data_date.year, last_data_date.month
        end_date = datetime(year, month, monthrange(year, month)[1])
    elif forecast_until == 'quarter_end':
        q_month = ((last_data_date.month - 1) // 3 + 1) * 3
        end_date = datetime(last_data_date.year, q_month, monthrange(last_data_date.year, q_month)[1])
    elif forecast_until == 'custom':
        end_date = last_data_date + timedelta(days=custom_days)
    else:
        end_date = datetime(last_data_date.year, 12, 31)

    future_dates = pd.date_range(start=last_data_date + timedelta(days=1), end=end_date)
    forecast_days = len(future_dates)
    if forecast_days <= 0:
        return pd.DataFrame(), last_data_date, 0, df_grouped

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
        forecast_vals = model.forecast(forecast_days)
        forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_vals})
    else:
        return pd.DataFrame(), last_data_date, 0, df_grouped

    forecast_full = pd.concat([
        df_grouped.rename(columns={'y': 'yhat'}),
        forecast
    ], ignore_index=True).sort_values('ds')

    forecast_full['yhat_lower'] = forecast_full['yhat'] * 0.95
    forecast_full['yhat_upper'] = forecast_full['yhat'] * 1.05

    return forecast, last_data_date, forecast_days, forecast_full

def calculate_target_analysis(df, forecast_df, last_date, target, mode):
    if mode == 'Monthly':
        current = df[(df['date'].dt.month == last_date.month) & (df['date'].dt.year == last_date.year)]['target'].sum()
    else:
        current = df[df['date'].dt.year == last_date.year]['target'].sum()

    forecast = forecast_df[forecast_df['ds'] > last_date]['yhat'].sum()
    total = current + forecast
    remaining = max(0, target - current)
    days_left = (forecast_df['ds'].max() - last_date).days
    per_day = round(remaining / days_left, 2) if days_left > 0 else 0
    pct = round((total / target) * 100, 2)

    return {
        "📌 Target": target,
        "🟢 Current Sales": round(current, 2),
        "🔮 Forecasted Sales (Remaining Days)": round(forecast, 2),
        "📊 Total Projected (Actual + Forecast)": round(total, 2),
        "📉 Remaining to Hit Target": round(remaining, 2),
        "📅 Days Left to Forecast": days_left,
        "📈 Required Per Day": per_day,
        "🎯 Projected % of Target": pct
    }

def get_forecast_explanation(method):
    return {
        "Prophet": "Prophet models trends and seasonality.",
        "Linear": "Linear regression fits a trend line.",
        "Exponential": "Exponential smoothing emphasizes recent data."
    }.get(method, "")
