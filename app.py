import streamlit as st
import pandas as pd
from forecast_utils import (
    preprocess_data, forecast_sales_prophet,
    plot_forecast, calculate_target_analysis,
    generate_recommendations, plot_daily_bar_chart,
    plot_actual_vs_forecast
)

from datetime import datetime

st.set_page_config(page_title="📊 Smart Sales Forecast App", layout="wide")
st.title("📊 Adaptive Sales Forecast & Target Tracker")

uploaded_file = st.file_uploader("📤 Upload Sales File (CSV or Excel)", type=["csv", "xlsx"])
data = None

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    raw_df.columns = (
        raw_df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    st.success("✅ File uploaded successfully!")
    st.write("📋 Detected columns:", raw_df.columns.tolist())

    if st.checkbox("👀 Want to see raw data?"):
        st.dataframe(raw_df.head())

    date_col = st.selectbox("📅 Select Date Column", raw_df.select_dtypes(include=['object', 'datetime']).columns)
    target_col = st.selectbox("🎯 Select Column to Forecast", raw_df.select_dtypes('number').columns)
    optional_filters = st.multiselect("🔍 Select Filters (Optional)", [col for col in raw_df.columns if col not in [date_col, target_col]])

    target_type = st.radio("📌 Choose Target Type", ["Monthly", "Yearly"])
    target_value = st.number_input("🎯 Enter Your Sales Target", step=1000)

    data = preprocess_data(raw_df, date_col, target_col, optional_filters)

if data is not None:
    st.sidebar.header("📌 Apply Filters")
    df = data.copy()

    for filt in data.columns:
        if filt not in ['date', 'target']:
            options = ["All"] + sorted(df[filt].dropna().unique().tolist())
            selected = st.sidebar.selectbox(f"Filter by {filt.capitalize()}", options, key=filt)
            if selected != "All":
                df = df[df[filt] == selected]

    # Forecast for remaining days of selected period
    forecast_df, days_left, end_date = forecast_sales_prophet(df, target_type)

    st.subheader("📊 Forecast Visualization")
    st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)

    st.subheader("📈 Actual vs Forecast")
    st.plotly_chart(plot_actual_vs_forecast(df, forecast_df), use_container_width=True)

    st.subheader("📊 Daily Sales Trend")
    st.plotly_chart(plot_daily_bar_chart(df), use_container_width=True)

    if target_value:
        analysis = calculate_target_analysis(df, forecast_df, target_value, target_type, days_left)

        st.markdown("---")
        st.subheader("📌 Target Analysis")
        st.metric("📌 Current Sales", f"{analysis['Current Sales']}")
        st.metric("🔮 Forecasted Sales (Remaining Days)", f"{analysis['Forecasted']}")
        st.metric("📊 Total Projected (Actual + Forecast)", f"{analysis['Projected Total']}")
        st.metric("📉 Remaining", f"{analysis['Remaining']}")
        st.metric("📅 Days Left", days_left)
        st.metric("📈 Required per Day", f"{analysis['Required per Day']}")
        st.metric("🎯 % of Target", f"{analysis['Projected % of Target']}%")

        st.success(generate_recommendations(analysis))
