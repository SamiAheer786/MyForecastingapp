import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
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

def forecast_sales(df, model_type, target_mode):
    df_grouped = df.groupby("date")["target"].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped = df_grouped.sort_values("ds")

    last_data_date = df_grouped['ds'].max()

    if target_mode == "Monthly":
        # Forecast till end of the same month as the last data
        year, month = last_data_date.year, last_data_date.month
        end_date = datetime(year, month, monthrange(year, month)[1])
    else:
        # Forecast till end of year
        end_date = datetime(last_data_date.year, 12, 31)

    forecast_days = (end_date - last_data_date).days
    if forecast_days <= 0:
        return pd.DataFrame(), last_data_date, 0  # No days to forecast
    if model_type == "Prophet":
    model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_grouped)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    return forecast, last_data_date, forecast_days

    elif model_type == "Linear":
        df_grouped['ds_ord'] = df_grouped['ds'].map(datetime.toordinal)
        m, b = polyfit(df_grouped['ds_ord'], df_grouped['y'], 1)
        future_dates = pd.date_range(start=last_data_date + timedelta(1), periods=forecast_days)
        forecast = pd.DataFrame({'ds': future_dates})
        forecast['yhat'] = [m * d.toordinal() + b for d in forecast['ds']]
        forecast['yhat_lower'] = forecast['yhat'] * 0.95
        forecast['yhat_upper'] = forecast['yhat'] * 1.05
        return forecast, last_data_date, forecast_days

    elif model_type == "Exponential":
        model = ExponentialSmoothing(df_grouped['y'], trend='add').fit()
        forecast_vals = model.forecast(forecast_days)
        future_dates = pd.date_range(start=last_data_date + timedelta(1), periods=forecast_days)
        forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_vals})
        forecast['yhat_lower'] = forecast['yhat'] * 0.9
        forecast['yhat_upper'] = forecast['yhat'] * 1.1
        return forecast, last_data_date, forecast_days

def calculate_target_analysis(df, forecast_df, last_date, target, mode):
    today = pd.Timestamp.today()
    if mode == 'Monthly':
        current = df[df['date'].dt.month == today.month]['target'].sum()
    else:
        current = df[df['date'].dt.year == today.year]['target'].sum()

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

def generate_recommendations(metrics):
    if metrics["🎯 Projected % of Target"] >= 100:
        return "✅ You're on track or exceeding your goal!"
    return f"⚠️ You need to sell {metrics['📈 Required Per Day']} units/day for {metrics['📅 Days Left to Forecast']} days."

def plot_forecast(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_upper'], name='Upper', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_lower'], name='Lower', line=dict(dash='dot')))
    fig.update_layout(title="📈 Forecast with Confidence Bands", xaxis_title="Date", yaxis_title="Sales")
    return fig

def plot_actual_vs_forecast(df, forecast_df):
    actual = df.groupby('date')['target'].sum().reset_index()
    actual.columns = ['ds', 'y']
    merged = pd.merge(forecast_df[['ds', 'yhat']], actual, on='ds', how='left')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], name='Actual'))
    fig.update_layout(title='📊 Actual vs Forecasted', xaxis_title='Date', yaxis_title='Sales')
    return fig

def plot_daily_bar_chart(df):
    daily = df.groupby('date')['target'].sum().reset_index()
    fig = px.bar(daily, x='date', y='target', title="📊 Daily Sales Trend")
    return fig

def generate_daily_table(forecast_df):
    return forecast_df[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'}).round(2)

def get_forecast_explanation(method):
    explanations = {
        "Prophet": "Prophet models seasonality and trends to forecast future values.",
        "Linear": "Linear regression fits a trend line using past data.",
        "Exponential": "Exponential smoothing gives more weight to recent observations."
    }
    return explanations.get(method, "No explanation available.")
