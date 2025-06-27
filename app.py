import streamlit as st
import pandas as pd
from datetime import datetime
from forecast_utils import (
    preprocess_data, forecast_sales,
    calculate_target_analysis, generate_recommendations,
    plot_forecast, plot_actual_vs_forecast,
    plot_daily_bar_chart
)

st.set_page_config(page_title="📊 Smart Forecasting App", layout="wide")
st.title("📊 Sales Forecast & Target Tracker (Smart BI)")

uploaded_file = st.file_uploader("📤 Upload Sales File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_").str.replace(r'[^\w\s]', '', regex=True)
    st.success("✅ File uploaded!")

    if st.checkbox("🔍 See Raw Data"):
        st.dataframe(df_raw.head())

    date_col = st.selectbox("📅 Select Date Column", df_raw.select_dtypes(include=["object", "datetime"]).columns)
    target_col = st.selectbox("🎯 Select Target Column", df_raw.select_dtypes("number").columns)
    group_cols = st.multiselect("🧩 Optional Group Filters (Region/Product/etc)", [col for col in df_raw.columns if col not in [date_col, target_col]])

    df_clean = preprocess_data(df_raw, date_col, target_col, group_cols)

    st.markdown("### 🧠 Choose Forecasting Method")
    forecast_method = st.radio("Forecast With", ["Prophet", "Linear Growth (basic)", "Exponential Smoothing"], horizontal=True)

    target_mode = st.radio("Target Type", ["Monthly", "Yearly"], horizontal=True)
    target_value = st.number_input("🎯 Enter Sales Target", step=1000)

    st.markdown("---")
    if st.button("🚀 Generate Forecast & Analysis"):
        forecast_df, days_left, last_date = forecast_sales(df_clean, forecast_method, target_mode)

        st.subheader("📊 Target Analysis")
        metrics = calculate_target_analysis(df_clean, forecast_df, target_value, days_left, target_mode)
        for k, v in metrics.items():
            st.metric(label=k, value=v)

        st.success(generate_recommendations(metrics))

        if st.button("📈 Show Charts"):
            st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)
            st.plotly_chart(plot_actual_vs_forecast(df_clean, forecast_df), use_container_width=True)
            st.plotly_chart(plot_daily_bar_chart(df_clean), use_container_width=True)

else:
    st.info("👋 Upload your dataset to begin.")
