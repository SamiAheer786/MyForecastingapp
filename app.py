import streamlit as st
import pandas as pd
from forecast_utils import (
    preprocess_data, forecast_sales,
    calculate_target_analysis, generate_recommendations,
    plot_forecast, plot_actual_vs_forecast,
    plot_daily_bar_chart, generate_daily_table
)

st.set_page_config(page_title="📊 Smart Sales Forecast App", layout="wide")
st.title("📊 Smart Sales Forecast & Target Tracker")

uploaded_file = st.file_uploader("📤 Upload Sales File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_").str.replace(r'[^\w\s]', '', regex=True)

    st.success("✅ File uploaded!")
    if st.checkbox("👁️ Show Data Head"):
        st.dataframe(df_raw.head())

    date_col = st.selectbox("📅 Select Date Column", df_raw.select_dtypes(include=["object", "datetime"]).columns)
    target_col = st.selectbox("🎯 Select Sales/Quantity Column", df_raw.select_dtypes("number").columns)
    filters = st.multiselect("🧩 Select Filter Columns (e.g., Region/Product)", [col for col in df_raw.columns if col not in [date_col, target_col]])

    df_clean = preprocess_data(df_raw, date_col, target_col, filters)

    st.markdown("## 🧠 Select Forecasting Method")
    model_choice = st.radio("Choose a method", ["Prophet", "Linear", "Exponential"])

    seasonal_effect = st.radio("📅 Any Special Seasonal Effects?", ["No", "Yes"])
    seasonal_dates = None
    if seasonal_effect == "Yes":
        st.warning("📌 Note: Data in selected dates may be treated as outliers or amplified.")
        seasonal_dates = st.date_input("Select special event dates", [], help="Like Eid, festivals, etc.")

    target_mode = st.radio("🎯 Target Period", ["Monthly", "Yearly"], horizontal=True)
    target_value = st.number_input("🔢 Enter Your Sales Target", step=1000)

    if st.button("🚀 Run Forecast"):
        forecast_df, last_data_date, days_left = forecast_sales(df_clean, model_choice, target_mode)
        metrics = calculate_target_analysis(df_clean, forecast_df, last_data_date, target_value, target_mode)

        st.subheader("📌 Target Analysis")
        for k, v in metrics.items():
            st.metric(label=k, value=v)

        st.success(generate_recommendations(metrics))

        if st.button("📈 Show Charts and Table"):
            st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)
            st.plotly_chart(plot_actual_vs_forecast(df_clean, forecast_df), use_container_width=True)
            st.plotly_chart(plot_daily_bar_chart(df_clean), use_container_width=True)
            st.subheader("📋 Daily Forecast Table")
            st.dataframe(generate_daily_table(forecast_df))
else:
    st.info("👋 Upload a sales data file to begin.")
