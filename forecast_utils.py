import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.express as px
from numpy import polyfit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from calendar import monthrange

# --- Utility Functions ---
def preprocess_data(df, date_col, target_col, filters=[]):
    df = df[[date_col, target_col] + filters].copy()
    df.columns = ['date', 'target'] + filters
    try:
        df['date'] = pd.to_datetime(df['date'])
    except:
        df['date'] = pd.to_datetime(df['date'].astype(str) + '-01', errors='coerce')
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

    forecast_days = len(future_dates)
    if forecast_days <= 0 or future_dates.min() <= df_grouped['ds'].max():
        print("⚠️ Forecast skipped: No future dates available.")
        return pd.DataFrame(), last_data_date, 0, df_grouped

    if model_type == "Prophet":
        model = Prophet()
        model.fit(df_grouped.rename(columns={"ds": "ds", "y": "y"}))
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

def plot_group_bar_chart(df, group_by):
    grouped = df.groupby(group_by)['target'].sum().reset_index()
    return px.bar(grouped, x=group_by, y='target', title=f"📊 Sales by {group_by.title()}")

# --- Streamlit App ---
st.set_page_config(page_title="📈 Multi-Dimensional Sales Forecast", layout="wide")
st.title("📈 Sales Forecast & Comparison Dashboard")

uploaded_file = st.file_uploader("📤 Upload Sales File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_").str.replace(r'[^\w\s]', '', regex=True)
    st.success("✅ File uploaded!")
    if st.checkbox("👁️ Show Data Preview"):
        st.dataframe(df_raw.head())

    date_col = st.selectbox("📅 Select Date Column", df_raw.columns)
    target_col = st.selectbox("🎯 Target Column", df_raw.select_dtypes(include="number").columns)
    filters = st.multiselect("🔎 Filter Columns (Optional)", [col for col in df_raw.columns if col not in [date_col, target_col]])

    df_clean = preprocess_data(df_raw, date_col, target_col, filters)

    model_choice = st.radio("🧠 Forecasting Model", ["Prophet", "Linear", "Exponential"])
    st.caption(get_forecast_explanation(model_choice))

    forecast_range = st.selectbox("⏳ Forecast Until", ["Till Month End", "Till Quarter End", "Till Year End", "Custom Days"])
    forecast_until = 'year_end'
    custom_days = None
    if forecast_range == "Till Month End":
        forecast_until = 'month_end'
    elif forecast_range == "Till Quarter End":
        forecast_until = 'quarter_end'
    elif forecast_range == "Custom Days":
        forecast_until = 'custom'
        custom_days = st.number_input("Days to Forecast", min_value=1, value=30)

    grouping_option = st.selectbox("📆 Group By", ["Daily", "Monthly", "Yearly"])

    if filters:
        comp_filter = st.selectbox("📍 Compare by Filter", filters)
        unique_vals = sorted(df_clean[comp_filter].dropna().unique())
        compare_button = st.button("🚀 Run Comparison Forecast")

        if compare_button:
            st.subheader("📊 Region/Product Forecast Comparison")
            for val in unique_vals:
                subset = df_clean[df_clean[comp_filter] == val]
                forecast, _, _, full_df = forecast_sales(subset, model_choice, forecast_until, custom_days, grouping_option)
                st.markdown(f"### 🔹 {comp_filter}: {val}")
                st.line_chart(full_df.set_index("ds")["yhat"])

    st.subheader("📥 Download Forecasted Data")
    download_btn = st.checkbox("✅ Generate for Export")
    if download_btn:
        full_df = df_clean.copy()
        forecast, _, _, forecast_full = forecast_sales(full_df, model_choice, forecast_until, custom_days, grouping_option)
        csv = forecast_full.to_csv(index=False).encode('utf-8')
        st.download_button("📁 Download Forecast CSV", data=csv, file_name="forecast_output.csv", mime='text/csv')

else:
    st.info("👋 Upload a CSV or Excel file to start.")
