import streamlit as st
import pandas as pd
from forecast_utils import preprocess_data, forecast_sales, get_forecast_explanation
import plotly.express as px

st.set_page_config(page_title="📈 Sales Forecast Dashboard", layout="wide")
st.title("📈 Sales Forecast & Comparison Tool")

uploaded_file = st.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_").str.replace(r'[^\w\s]', '', regex=True)
    
    st.success("✅ File uploaded!")
    if st.checkbox("👁️ Preview Data"):
        st.dataframe(df_raw.head())

    date_col = st.selectbox("📅 Select Date Column", df_raw.select_dtypes(include=["object", "datetime"]).columns)
    target_col = st.selectbox("🎯 Select Target Column", df_raw.select_dtypes(include="number").columns)
    filters = st.multiselect("🔎 Optional Filters", [col for col in df_raw.columns if col not in [date_col, target_col]])

    df_clean = preprocess_data(df_raw, date_col, target_col, filters)

    model_choice = st.radio("🧠 Forecasting Method", ["Prophet", "Linear", "Exponential"])
    st.caption(get_forecast_explanation(model_choice))

    horizon = st.selectbox("⏳ Forecast Horizon", ["Till Month End", "Till Quarter End", "Till Year End", "Custom Days"])
    forecast_until = 'year_end'
    custom_days = None
    if horizon == "Till Month End":
        forecast_until = 'month_end'
    elif horizon == "Till Quarter End":
        forecast_until = 'quarter_end'
    elif horizon == "Custom Days":
        forecast_until = 'custom'
        custom_days = st.number_input("📆 Enter Days", min_value=1, value=30)

    group_by = st.selectbox("📆 Grouping Level", ["Daily", "Monthly", "Yearly"])

    if filters:
        compare_filter = st.selectbox("📍 Compare by", filters)
        unique_vals = sorted(df_clean[compare_filter].dropna().unique())
        if st.button("🚀 Run Group Forecast"):
            st.subheader("📊 Forecast Comparison")
            for val in unique_vals:
                subset = df_clean[df_clean[compare_filter] == val]
                forecast, _, _, forecast_full = forecast_sales(subset, model_choice, forecast_until, custom_days, group_by)
                st.markdown(f"### 🔹 {compare_filter}: {val}")
                st.line_chart(forecast_full.set_index('ds')['yhat'])

    if st.checkbox("📥 Download Forecast Data"):
        forecast, _, _, full_forecast = forecast_sales(df_clean, model_choice, forecast_until, custom_days, group_by)
        csv = full_forecast.to_csv(index=False).encode('utf-8')
        st.download_button("📁 Download CSV", data=csv, file_name="forecast_output.csv", mime="text/csv")

else:
    st.info("📁 Please upload your dataset to begin.")
