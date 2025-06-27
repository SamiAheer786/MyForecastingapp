import pandas as pd
from prophet import Prophet
from datetime import datetime
import plotly.graph_objects as go
import streamlit as st

def preprocess_data(df, date_col, target_col, filter_cols=[]):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[[date_col, target_col] + filter_cols]
    df.columns = ['date', 'target'] + filter_cols
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['date', 'target'], inplace=True)
    return df

def forecast_sales_prophet(df):
    daily = df.groupby('date')['target'].sum().reset_index()
    daily.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(daily)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], model

def forecast_sales_linear(df):
    daily = df.groupby('date')['target'].sum().reset_index()
    daily.columns = ['ds', 'y']
    daily['yhat'] = daily['y'].expanding().mean()
    future_dates = pd.date_range(start=daily['ds'].max() + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({'ds': future_dates, 'yhat': [daily['yhat'].iloc[-1]] * 30})
    forecast_df = pd.concat([daily[['ds', 'yhat']], future_df], ignore_index=True)
    forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.95
    forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.05
    return forecast_df

def plot_forecast(forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(title='📈 Forecasted Sales', xaxis_title='Date', yaxis_title='Quantity')
    return fig

def calculate_target_analysis_updated(df, forecast_df, target_value, target_type):
    today = pd.to_datetime(datetime.today().date())
    current = df[df['date'].dt.month == today.month]['target'].sum() if target_type == 'Monthly' else df['target'].sum()
    forecast = forecast_df[forecast_df['ds'] >= today]['yhat'].sum()
    total = current + forecast
    remaining = max(0, target_value - current)
    days_left = max(1, (forecast_df['ds'].max() - today).days)
    per_day = round(remaining / days_left, 2)
    pct = round((total / target_value) * 100, 2)
    return {
        "Target": target_value,
        "Current Sales": round(current, 2),
        "Forecasted": round(forecast, 2),
        "Projected Total": round(total, 2),
        "Remaining": round(remaining, 2),
        "Required per Day": per_day,
        "Projected % of Target": pct,
        "Days Left": days_left
    }

def generate_recommendations(analysis):
    if analysis['Projected % of Target'] >= 100:
        return "✅ You're on track to meet or exceed your target!"
    return f"🚀 You need to sell {analysis['Required per Day']} units/day to hit your goal."
