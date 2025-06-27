import streamlit as st
import pandas as pd
from forecast_utils import (
    preprocess_data, forecast_sales_prophet, forecast_sales_linear,
    plot_forecast, calculate_target_analysis_updated, generate_recommendations
)

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

    date_col = st.selectbox("📅 Select Date Column", raw_df.select_dtypes(include=['datetime64', 'object']).columns)
    target_col = st.selectbox("🎯 Select Column to Forecast", raw_df.select_dtypes('number').columns)
    optional_filters = st.multiselect("🔍 Select Filters (Optional)", [col for col in raw_df.columns if col not in [date_col, target_col]])

    forecast_method = st.selectbox("📈 Choose Forecasting Method", ["Prophet", "Linear Growth"])

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

    if forecast_method == "Prophet":
        forecast_df, model = forecast_sales_prophet(df)
    else:
        forecast_df = forecast_sales_linear(df)

    st.subheader("📊 Forecast Visualization")
    st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Target Analysis")
    target_type = st.radio("Target Type", ["Monthly", "Yearly"])
    target_value = st.number_input("Enter Sales Target", step=100)

    if target_value:
        analysis = calculate_target_analysis_updated(df, forecast_df, target_value, target_type)
        st.metric(label="📌 Projected Total", value=f"{analysis['Projected Total']}")
        st.metric(label="📊 % of Target", value=f"{analysis['Projected % of Target']}%")
        st.write(analysis)
        st.success(generate_recommendations(analysis))
else:
    st.info("👋 Upload a file and start by selecting columns.")
