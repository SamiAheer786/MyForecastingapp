import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from calendar import monthrange

def preprocess_data(df, date_col, target_col, group_cols=[]):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[[date_col, target_col] + group_cols]
    df.columns = ['date', 'target'] + group_cols
    df.dropna(subset=['date', 'target'], inplace=True)
    return df

def forecast_sales(df, method, target_mode):
    df = df.copy()
    df = df.groupby("date")["target"].sum().reset_index()
    df.columns = ['ds', 'y']
    df = df[df['ds'] < datetime.today()]  # Trim for forecast

    today = pd.Timestamp.today()
    if target_mode == "Monthly":
        last_day = pd.Timestamp(today.year, today.month, monthrange(today.year, today.month)[1])
    else:
        last_day = pd.Timestamp(today.year, 12, 31)

    days_left = (last_day - today).days

    if method == "Prophet":
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=days_left)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], days_left, last_day

    elif method == "Linear Growth (basic)":
        from numpy import polyfit
        df['ds_ordinal'] = df['ds'].map(datetime.toordinal)
        m, b = polyfit(df['ds_ordinal'], df['y'], 1)
        future_dates = pd.date_range(start=today + timedelta(1), end=last_day)
        forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': [m * d.toordinal() + b for d in future_dates]
        })
        forecast['yhat_lower'] = forecast['yhat'] * 0.95
        forecast['yhat_upper'] = forecast['yhat'] * 1.05
        return forecast, days_left, last_day

    elif method == "Exponential Smoothing":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(df['y'], trend="add", seasonal=None).fit()
        forecast_vals = model.forecast(days_left)
        future_dates = pd.date_range(start=today + timedelta(1), end=last_day)
        forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_vals})
        forecast['yhat_lower'] = forecast['yhat'] * 0.9
        forecast['yhat_upper'] = forecast['yhat'] * 1.1
        return forecast, days_left, last_day

def calculate_target_analysis(df, forecast_df, target, days_left, mode):
    today = pd.Timestamp.today()
    if mode == 'Monthly':
        current = df[df['date'].dt.month == today.month]['target'].sum()
    else:
        current = df[df['date'].dt.year == today.year]['target'].sum()

    forecast = forecast_df[forecast_df['ds'] > today]['yhat'].sum()
    total = current + forecast
    remaining = max(0, target - current)
    per_day = round(remaining / days_left, 2) if days_left > 0 else 0
    pct = round((total / target) * 100, 2) if target > 0 else 0

    return {
        "📌 Target": target,
        "🟢 Current Sales": round(current, 2),
        "🔮 Forecasted Sales (Remaining Days)": round(forecast, 2),
        "📊 Total Projected (Current + Forecast)": round(total, 2),
        "📉 Remaining to Hit Target": round(remaining, 2),
        "📅 Days Left": days_left,
        "📈 Required per Day": per_day,
        "🎯 Projected % of Target": pct
    }

def generate_recommendations(analysis):
    if analysis["🎯 Projected % of Target"] >= 100:
        return "✅ You're already above your goal. Great job!"
    return f"⚠️ To meet your target, sell {analysis['📈 Required per Day']} units/day for the remaining {analysis['📅 Days Left']} days."

def plot_forecast(forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
    fig.update_layout(title='📈 Forecasted Sales with Confidence Bands', xaxis_title='Date', yaxis_title='Sales')
    return fig

def plot_actual_vs_forecast(df, forecast_df):
    actual = df.groupby('date')['target'].sum().reset_index()
    actual.columns = ['ds', 'y']
    forecast_trim = forecast_df[['ds', 'yhat']]
    merged = pd.merge(forecast_trim, actual, on='ds', how='left')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], name='Actual'))
    fig.update_layout(title='📊 Actual vs Forecasted Sales', xaxis_title='Date', yaxis_title='Sales')
    return fig

def plot_daily_bar_chart(df):
    df_daily = df.groupby('date')['target'].sum().reset_index()
    fig = px.bar(df_daily, x='date', y='target', title='📊 Daily Sales Volume')
    return fig
