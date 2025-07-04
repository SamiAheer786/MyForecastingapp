import os
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import joblib
from numpy import polyfit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from calendar import monthrange
import plotly.graph_objects as go
import plotly.express as px

MODEL_PATH = "model.pkl"
DATA_PATH = "historical_data.csv"

def preprocess_data(df, date_col, target_col):
    df = df[[date_col, target_col]].copy()
    df.columns = ['date', 'target']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(inplace=True)
    return df

def append_and_train(new_data, append=True):
    if append and os.path.exists(DATA_PATH):
        old_data = pd.read_csv(DATA_PATH, parse_dates=['date'])
        full_data = pd.concat([old_data, new_data]).drop_duplicates(subset='date')
    else:
        full_data = new_data

    full_data.sort_values('date', inplace=True)
    full_data.to_csv(DATA_PATH, index=False)

    prophet_df = full_data.groupby('date')['target'].sum().reset_index()
    prophet_df.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    joblib.dump(model, MODEL_PATH)

    return model, full_data

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        data = pd.read_csv(DATA_PATH, parse_dates=['date'])
        return model, data
    else:
        return None, None

def forecast_sales(model, last_date, target_mode):
    if target_mode == "Monthly":
        end_date = datetime(last_date.year, last_date.month, monthrange(last_date.year, last_date.month)[1])
    else:
        end_date = datetime(last_date.year, 12, 31)

    forecast_days = (end_date - last_date).days
    if forecast_days <= 0:
        return pd.DataFrame(), 0

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], forecast_days

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

def plot_daily_bar_chart(df):
    daily = df.groupby('date')['target'].sum().reset_index()
    fig = px.bar(daily, x='date', y='target', title="📊 Daily Sales Trend")
    return fig

def generate_daily_table(forecast_df):
    return forecast_df[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'}).round(2)

