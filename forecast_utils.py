import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

def preprocess_data(df, date_col, target_col, filter_cols=[]):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[[date_col, target_col] + filter_cols]
    df.columns = ['date', 'target'] + filter_cols
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['date', 'target'], inplace=True)
    return df

def forecast_sales_prophet(df, target_type):
    daily = df.groupby('date')['target'].sum().reset_index()
    daily.columns = ['ds', 'y']

    today = pd.to_datetime(datetime.today().date())

    if target_type == 'Monthly':
        end_day = datetime(today.year, today.month + 1, 1) - timedelta(days=1) if today.month != 12 else datetime(today.year, 12, 31)
    else:
        end_day = datetime(today.year, 12, 31)

    days_left = (end_day - today).days

    model = Prophet(daily_seasonality=True)
    model.fit(daily)
    future = model.make_future_dataframe(periods=days_left)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], days_left, end_day

def calculate_target_analysis(df, forecast_df, target_value, target_type, days_left):
    today = pd.to_datetime(datetime.today().date())

    current = df[df['date'].dt.month == today.month]['target'].sum() if target_type == 'Monthly' else df[df['date'].dt.year == today.year]['target'].sum()
    forecast = forecast_df[forecast_df['ds'] > today]['yhat'].sum()

    total = current + forecast
    remaining = max(0, target_value - current)
    per_day = round(remaining / days_left, 2) if days_left > 0 else 0
    pct = round((total / target_value) * 100, 2)

    return {
        "Target": target_value,
        "Current Sales": round(current, 2),
        "Forecasted": round(forecast, 2),
        "Projected Total": round(total, 2),
        "Remaining": round(remaining, 2),
        "Required per Day": per_day,
        "Projected % of Target": pct
    }

def generate_recommendations(analysis):
    if analysis['Projected % of Target'] >= 100:
        return "✅ You're on track to meet or exceed your target!"
    return f"🚀 You need to sell {analysis['Required per Day']} units/day to hit your goal."

def plot_forecast(forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(title='📈 Forecasted Sales', xaxis_title='Date', yaxis_title='Quantity')
    return fig

def plot_actual_vs_forecast(df, forecast_df):
    actual = df.groupby('date')['target'].sum().reset_index()
    actual.columns = ['ds', 'y']

    forecast_trimmed = forecast_df[['ds', 'yhat']]
    merged = pd.merge(forecast_trimmed, actual, on='ds', how='left')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], name='Actual'))
    fig.update_layout(title='📊 Actual vs Forecast', xaxis_title='Date', yaxis_title='Quantity')
    return fig

def plot_daily_bar_chart(df):
    df_daily = df.groupby('date')['target'].sum().reset_index()
    fig = px.bar(df_daily, x='date', y='target', title='📊 Daily Sales Volume')
    return fig
